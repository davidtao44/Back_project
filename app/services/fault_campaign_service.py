"""Servicio para ejecutar campañas de fallos en redes neuronales.

Incluye la clase FaultCampaign (antes en fault_campaign.py) y funciones
orquestadoras para los endpoints de ruta.
"""

import io
import json
import os
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from fastapi import HTTPException
from PIL import Image as PILImage
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score

from app.schemas.fault import FaultCampaignRequest, WeightFaultCampaignRequest
from app.utils.json_utils import sanitize_for_json
from fault_injection.manual_inference import ManualInference
from fault_injection.weight_fault_injector import WeightFaultInjector

# ── Job store (in-memory) ─────────────────────────────────────────────────────
_job_store: Dict[str, Dict] = {}
_job_lock = threading.Lock()


def _update_job(job_id: str, **kwargs) -> None:
    with _job_lock:
        if job_id in _job_store:
            _job_store[job_id].update(kwargs)


def get_job_status(job_id: str) -> Optional[Dict]:
    with _job_lock:
        job = _job_store.get(job_id)
    if not job:
        return None
    return {
        "job_id": job_id,
        "status": job["status"],        # pending | running | done | error
        "progress": job["progress"],    # 0-100
        "phase": job["phase"],
        "error": job.get("error"),
    }


def get_job_results(job_id: str) -> Optional[Dict]:
    with _job_lock:
        job = _job_store.get(job_id)
    if not job:
        return None
    return job.get("results")


class FaultCampaign:
    """Ejecuta campañas de fallos y compara métricas golden vs con fallos."""

    def __init__(self, model_path: str, image_dir: Optional[str] = None, session_id: Optional[str] = None):
        self.model_path = model_path
        self.image_dir = image_dir
        self.session_id = session_id or f"campaign_{int(time.time())}_{str(uuid.uuid4())[:8]}"

        self.model = tf.keras.models.load_model(model_path)
        print(f"✅ Modelo cargado: {model_path}")

        self.images, self.labels = self._load_dataset()
        print(f"📊 Dataset cargado: {len(self.images)} imágenes")

        self.manual_inference = ManualInference(model_instance=self.model)
        self.weight_fault_injector = WeightFaultInjector()

        self.results = {
            "golden": {"predictions": [], "labels": []},
            "fault": {"predictions": [], "labels": []},
            "metrics": {},
        }

    def _load_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.image_dir and os.path.exists(self.image_dir):
            return self._load_images_from_directory()
        return self._load_mnist_dataset()

    def _load_images_from_directory(self) -> Tuple[np.ndarray, np.ndarray]:
        """Carga imágenes desde image_dir/0/, image_dir/1/, ..., image_dir/9/."""
        images = []
        labels = []

        for digit_folder in os.listdir(self.image_dir):
            digit_path = os.path.join(self.image_dir, digit_folder)

            if os.path.isdir(digit_path) and digit_folder.isdigit():
                label = int(digit_folder)
                print(f"📁 Cargando imágenes del dígito {label}...")

                for filename in os.listdir(digit_path):
                    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                        try:
                            image_path = os.path.join(digit_path, filename)
                            img = PILImage.open(image_path).convert("L")
                            img = img.resize((28, 28))
                            img_array = np.array(img) / 255.0

                            images.append(img_array)
                            labels.append(label)
                        except Exception as e:
                            print(f"⚠️ No se pudo procesar archivo: {filename} - {str(e)}")
                            continue

        if not images:
            print("⚠️ No se encontraron imágenes válidas, usando MNIST")
            return self._load_mnist_dataset()

        images_array = np.array(images)
        labels_array = np.array(labels)

        if len(images_array.shape) == 3:
            images_array = np.expand_dims(images_array, axis=-1)

        print(f"✅ Cargadas {len(images_array)} imágenes locales con forma: {images_array.shape}")
        return images_array, labels_array

    def _load_mnist_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_test = x_test.astype("float32") / 255.0
        if len(x_test.shape) == 3:
            x_test = np.expand_dims(x_test, axis=-1)
        return x_test, y_test

    def _convert_image_to_bytes(self, image: np.ndarray) -> bytes:
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)

        image_pil = PILImage.fromarray((image.squeeze() * 255).astype(np.uint8), mode="L")

        img_byte_arr = io.BytesIO()
        image_pil.save(img_byte_arr, format="PNG")
        return img_byte_arr.getvalue()

    def _perform_inference_on_samples(self, indices: np.ndarray, description: str) -> Tuple[List[int], List[int]]:
        predictions = []
        labels = []

        if "con pesos modificados" in description:
            for layer in self.model.layers:
                if hasattr(layer, "kernel") and layer.kernel is not None:
                    weights = layer.kernel.numpy()
                    print(f"🔍 DEBUG: Pesos de {layer.name} durante inferencia con fallos - muestra: {weights.flat[:3]}")
                    break

        for i, idx in enumerate(indices):
            image = self.images[idx]
            label = self.labels[idx]

            image_bytes = self._convert_image_to_bytes(image)
            result = self.manual_inference.perform_manual_inference(image_bytes)
            predicted_class = result["final_prediction"]["predicted_class"]

            predictions.append(predicted_class)
            labels.append(label)

            if i < 3:
                print(
                    f"  🔍 DEBUG {description} - Muestra {i + 1}: Predicción={predicted_class}, "
                    f"Etiqueta={label}, Índice={idx}"
                )

            if (i + 1) % 5 == 0:
                print(
                    f"  Procesadas {i + 1}/{len(indices)} muestras {description} - "
                    f"Última predicción: {predicted_class}, Etiqueta real: {label}"
                )

        return predictions, labels

    def _create_campaign_results(
        self,
        golden_predictions: List[int],
        golden_labels: List[int],
        fault_predictions: List[int],
        fault_labels: List[int],
        num_samples: int,
        execution_time: float,
        config: Dict[str, Any],
        config_key: str,
    ) -> Dict[str, Any]:
        print(
            f"🔍 DEBUG: Calculando métricas golden con {len(golden_labels)} etiquetas "
            f"y {len(golden_predictions)} predicciones"
        )
        golden_metrics = self.calculate_metrics(golden_labels, golden_predictions)
        print(f"🔍 DEBUG: Métricas golden calculadas: {golden_metrics}")

        print(
            f"🔍 DEBUG: Calculando métricas con fallos con {len(fault_labels)} etiquetas "
            f"y {len(fault_predictions)} predicciones"
        )
        fault_metrics = self.calculate_metrics(fault_labels, fault_predictions)
        print(f"🔍 DEBUG: Métricas con fallos calculadas: {fault_metrics}")

        comparison = self._compare_predictions(golden_predictions, fault_predictions)
        print(f"🔍 DEBUG: Comparación calculada: {comparison}")

        results = {
            "golden_results": {
                "predictions": golden_predictions,
                "labels": golden_labels,
                "metrics": golden_metrics,
            },
            "fault_results": {
                "predictions": fault_predictions,
                "labels": fault_labels,
                "metrics": fault_metrics,
            },
            "comparison": comparison,
            "campaign_info": {
                "session_id": self.session_id,
                "model_path": self.model_path,
                "num_samples": num_samples,
                "execution_time_seconds": execution_time,
                config_key: config,
            },
        }

        print("🔍 DEBUG: Estructura final de resultados:")
        print(f"  - Golden metrics keys: {list(golden_metrics.keys()) if golden_metrics else 'None'}")
        print(f"  - Fault metrics keys: {list(fault_metrics.keys()) if fault_metrics else 'None'}")
        print(f"  - Comparison keys: {list(comparison.keys()) if comparison else 'None'}")

        return results

    def run_golden_inference(self, num_samples: int) -> Tuple[List[int], List[int]]:
        print(f"🏆 Iniciando inferencia golden con {num_samples} muestras...")

        self.selected_indices = np.random.choice(
            len(self.images), size=min(num_samples, len(self.images)), replace=False
        )
        print(f"🔍 DEBUG: Índices seleccionados para golden: {self.selected_indices[:5]}...")

        predictions, labels = self._perform_inference_on_samples(self.selected_indices, "golden")

        self.results["golden"]["predictions"] = predictions
        self.results["golden"]["labels"] = labels

        print(f"✅ Inferencia golden completada: {len(predictions)} predicciones")
        return predictions, labels

    def run_fault_inference(self, num_samples: int, fault_config: Dict[str, Any]) -> Tuple[List[int], List[int]]:
        print("⚡ Iniciando inferencia con fallos en activaciones...")

        self.manual_inference.configure_fault_injection(fault_config)

        predictions = []
        labels = []

        indices = np.random.choice(len(self.images), size=min(num_samples, len(self.images)), replace=False)

        for i, idx in enumerate(indices):
            image = self.images[idx]
            label = self.labels[idx]

            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)

            result = self.manual_inference.perform_inference_with_faults(image)
            predicted_class = result["predicted_class"]

            predictions.append(predicted_class)
            labels.append(label)

            if (i + 1) % 10 == 0:
                print(f"  Procesadas {i + 1}/{len(indices)} muestras con fallos")

        self.results["fault"]["predictions"] = predictions
        self.results["fault"]["labels"] = labels

        print(f"✅ Inferencia con fallos completada: {len(predictions)} predicciones")
        return predictions, labels

    def run_weight_fault_campaign(
        self,
        num_samples: int,
        weight_fault_config: Dict[str, Any],
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> Dict[str, Any]:
        def cb(pct: int, phase: str):
            if progress_callback:
                progress_callback(pct, phase)

        print("🎯 Iniciando campaña de fallos en pesos...")
        start_time = time.time()

        cb(5, "Cargando imágenes y modelo")
        golden_predictions, golden_labels = self.run_golden_inference(num_samples)
        cb(35, "Inferencia golden completada")

        print("💾 Haciendo backup de pesos originales...")
        self.weight_fault_injector.backup_original_weights(self.model)
        cb(40, "Configurando inyección de fallos en pesos")

        print("🔧 Configurando inyección de fallos en pesos...")
        for layer_name, layer_config in weight_fault_config.get("layers", {}).items():
            self.weight_fault_injector.configure_fault(layer_name, layer_config)

        print("⚡ Aplicando fallos en pesos del modelo...")
        injected_weight_faults = self.weight_fault_injector.inject_faults_in_weights(self.model)
        print(f"✅ Inyectados {len(injected_weight_faults)} fallos en pesos")
        cb(50, "Inyectando fallos y ejecutando inferencia")

        print("🔍 Ejecutando inferencia con pesos modificados...")
        if not hasattr(self, "selected_indices"):
            raise ValueError("❌ ERROR: No se han seleccionado índices en la inferencia golden")
        fault_predictions, fault_labels = self._perform_inference_on_samples(
            self.selected_indices, "con pesos modificados"
        )
        cb(80, "Inferencia con fallos completada")

        print("🔄 Restaurando pesos originales...")
        self.weight_fault_injector.restore_original_weights(self.model)

        differences = [i for i in range(len(golden_predictions)) if golden_predictions[i] != fault_predictions[i]]
        print(f"🔍 DEBUG: Total de diferencias: {len(differences)}/{len(golden_predictions)}")
        cb(90, "Calculando métricas y resultados")

        execution_time = time.time() - start_time
        results = self._create_campaign_results(
            golden_predictions,
            golden_labels,
            fault_predictions,
            fault_labels,
            num_samples,
            execution_time,
            weight_fault_config,
            "weight_fault_config",
        )

        cb(100, "¡Campaña completada!")
        print(f"✅ Campaña de fallos en pesos completada en {execution_time:.2f} segundos")
        return results

    def run_campaign(self, num_samples: int, fault_config: Dict[str, Any]) -> Dict[str, Any]:
        print("🎯 Iniciando campaña de fallos en activaciones...")
        start_time = time.time()

        golden_predictions, golden_labels = self.run_golden_inference(num_samples)
        fault_predictions, fault_labels = self.run_fault_inference(num_samples, fault_config)

        execution_time = time.time() - start_time
        results = self._create_campaign_results(
            golden_predictions,
            golden_labels,
            fault_predictions,
            fault_labels,
            num_samples,
            execution_time,
            fault_config,
            "fault_config",
        )

        print(f"✅ Campaña de fallos completada en {execution_time:.2f} segundos")
        return results

    def calculate_metrics(self, true_labels: List[int], predictions: List[int]) -> Dict[str, Any]:
        y_true = np.array(true_labels)
        y_pred = np.array(predictions)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        return {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "confusion_matrix": conf_matrix.tolist(),
            "classification_report": class_report,
            "num_samples": len(true_labels),
            "correct_predictions": int(np.sum(y_true == y_pred)),
            "incorrect_predictions": int(np.sum(y_true != y_pred)),
        }

    def _compare_predictions(
        self, golden_predictions: List[int], fault_predictions: List[int]
    ) -> Dict[str, Any]:
        golden_array = np.array(golden_predictions)
        fault_array = np.array(fault_predictions)

        same_predictions = np.sum(golden_array == fault_array)
        different_predictions = np.sum(golden_array != fault_array)
        different_indices = np.where(golden_array != fault_array)[0].tolist()

        return {
            "samples_with_same_predictions": int(same_predictions),
            "samples_with_different_predictions": int(different_predictions),
            "percentage_different": float(different_predictions / len(golden_predictions) * 100),
            "different_prediction_indices": different_indices,
        }

    def save_results(self, results: Dict[str, Any], output_file: str = None) -> str:
        if output_file is None:
            output_file = f"fault_campaign_results_{self.session_id}.json"

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"💾 Resultados guardados en: {output_file}")
        return output_file


def run_activation_fault_campaign(request: FaultCampaignRequest, current_user: dict):
    """Orquesta la ejecución de una campaña de fallos en activaciones."""
    try:
        print(f"🎯 Iniciando campaña de fallos para usuario: {current_user.get('username', 'unknown')}")

        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=404, detail=f"Modelo no encontrado: {request.model_path}")

        campaign = FaultCampaign(model_path=request.model_path, image_dir=request.image_dir)
        results = campaign.run_campaign(num_samples=request.num_samples, fault_config=request.fault_config)

        print("✅ Campaña de fallos completada exitosamente")

        return {
            "success": True,
            "message": "Campaña de fallos ejecutada exitosamente",
            "results": results,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error en campaña de fallos: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error ejecutando campaña de fallos: {str(e)}")


def run_weight_fault_campaign_request(request: WeightFaultCampaignRequest, current_user: dict):
    """Orquesta la ejecución de una campaña de fallos en pesos."""
    try:
        print(
            f"🎯 Iniciando campaña de fallos en pesos para usuario: {current_user.get('username', 'unknown')}"
        )

        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=404, detail=f"Modelo no encontrado: {request.model_path}")

        campaign = FaultCampaign(model_path=request.model_path, image_dir=request.image_dir)
        results = campaign.run_weight_fault_campaign(
            num_samples=request.num_samples,
            weight_fault_config=request.weight_fault_config,
        )

        print("✅ Campaña de fallos en pesos completada exitosamente")

        return {
            "success": True,
            "message": "Campaña de fallos en pesos ejecutada exitosamente",
            "results": sanitize_for_json(results),
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error en campaña de fallos en pesos: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error ejecutando campaña de fallos en pesos: {str(e)}"
        )


def start_weight_fault_campaign_job(request: WeightFaultCampaignRequest, current_user: dict) -> str:
    """Lanza la campaña en un hilo separado y devuelve el job_id inmediatamente."""
    job_id = str(uuid.uuid4())

    with _job_lock:
        _job_store[job_id] = {
            "status": "pending",
            "progress": 0,
            "phase": "En cola...",
            "results": None,
            "error": None,
        }

    def _run():
        _update_job(job_id, status="running", progress=0, phase="Iniciando campaña...")
        try:
            if not os.path.exists(request.model_path):
                raise ValueError(f"Modelo no encontrado: {request.model_path}")

            def on_progress(pct: int, phase: str):
                _update_job(job_id, progress=pct, phase=phase)

            campaign = FaultCampaign(model_path=request.model_path, image_dir=request.image_dir)
            results = campaign.run_weight_fault_campaign(
                num_samples=request.num_samples,
                weight_fault_config=request.weight_fault_config,
                progress_callback=on_progress,
            )
            _update_job(
                job_id,
                status="done",
                progress=100,
                phase="¡Campaña completada!",
                results={
                    "success": True,
                    "message": "Campaña de fallos en pesos ejecutada exitosamente",
                    "results": sanitize_for_json(results),
                },
            )
        except Exception as e:
            print(f"❌ Error en job {job_id}: {str(e)}")
            _update_job(job_id, status="error", phase="Error", error=str(e))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return job_id
