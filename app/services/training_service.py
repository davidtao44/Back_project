"""Servicio de entrenamiento de CNN como jobs en segundo plano.

Construye o carga un modelo, lo entrena sobre un dataset integrado de Keras y
lo guarda en models/ junto a un sidecar .meta.json. El entrenamiento corre en
un hilo aparte; el progreso por época se consulta vía get_training_status.
"""

import json
import threading
import time
import uuid
from typing import Dict, Optional

import tensorflow as tf

from app.services.cnn_builder_service import build_model, save_built_model
from app.services.dataset_service import DATASETS, load_dataset

# ── Job store en memoria ──────────────────────────────────────────────────────
_training_jobs: Dict[str, Dict] = {}
_jobs_lock = threading.Lock()


def _update_job(job_id: str, **kwargs) -> None:
    with _jobs_lock:
        if job_id in _training_jobs:
            _training_jobs[job_id].update(kwargs)


def _append_history(job_id: str, entry: Dict) -> None:
    with _jobs_lock:
        if job_id in _training_jobs:
            _training_jobs[job_id]["history"].append(entry)


def get_training_status(job_id: str) -> Optional[Dict]:
    with _jobs_lock:
        job = _training_jobs.get(job_id)
        if not job:
            return None
        return {
            "job_id": job_id,
            "status": job["status"],
            "progress": job["progress"],
            "phase": job["phase"],
            "history": list(job["history"]),
            "error": job.get("error"),
        }


def get_training_result(job_id: str) -> Optional[Dict]:
    with _jobs_lock:
        job = _training_jobs.get(job_id)
        return job.get("results") if job else None


# ── Callback de progreso ──────────────────────────────────────────────────────
class _JobProgressCallback(tf.keras.callbacks.Callback):
    """Empuja métricas por época al job store."""

    def __init__(self, job_id: str, total_epochs: int):
        super().__init__()
        self.job_id = job_id
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        entry = {
            "epoch": epoch + 1,
            "loss": float(logs.get("loss", 0.0)),
            "accuracy": float(logs.get("accuracy", 0.0)),
            "val_loss": float(logs.get("val_loss", 0.0)),
            "val_accuracy": float(logs.get("val_accuracy", 0.0)),
        }
        _append_history(self.job_id, entry)
        _update_job(
            self.job_id,
            progress=int((epoch + 1) / self.total_epochs * 100),
            phase=f"Entrenando — época {epoch + 1}/{self.total_epochs}",
        )


def _make_optimizer(name: str, learning_rate: float):
    """Crear un optimizador de Keras a partir de su nombre."""
    name = (name or "adam").lower()
    optimizers = {
        "adam": tf.keras.optimizers.Adam,
        "sgd": tf.keras.optimizers.SGD,
        "rmsprop": tf.keras.optimizers.RMSprop,
    }
    return optimizers.get(name, tf.keras.optimizers.Adam)(learning_rate=learning_rate)


def _resolve_model(request) -> tf.keras.Model:
    """Construir el modelo desde un spec o cargarlo desde una ruta existente."""
    if request.model_spec is not None:
        spec = request.model_spec
        spec_dict = spec.model_dump() if hasattr(spec, "model_dump") else spec.dict()
        return build_model(spec_dict)
    if request.model_path:
        return tf.keras.models.load_model(request.model_path)
    raise ValueError("Debe indicar model_spec o model_path")


def _validate_compatibility(model: tf.keras.Model, dataset: str) -> None:
    """Verificar que la entrada del modelo coincide con la del dataset."""
    meta = DATASETS[dataset]
    model_shape = list(model.input_shape[1:])
    if model_shape != meta["input_shape"]:
        raise ValueError(
            f"La entrada del modelo {model_shape} no coincide con la del dataset "
            f"'{dataset}' {meta['input_shape']}. Ajusta la arquitectura."
        )


def _run_training(job_id: str, request) -> None:
    """Flujo completo de entrenamiento (se ejecuta en un hilo aparte)."""
    try:
        if request.dataset not in DATASETS:
            raise ValueError(f"Dataset desconocido: '{request.dataset}'")

        # 1. Preparar el modelo
        _update_job(job_id, status="running", phase="Preparando el modelo", progress=2)
        model = _resolve_model(request)
        _validate_compatibility(model, request.dataset)

        # 2. Cargar el dataset (descarga la primera vez)
        _update_job(job_id, phase="Descargando/cargando dataset", progress=5)
        (x_train, y_train), (x_test, y_test) = load_dataset(
            request.dataset, limit=request.train_limit
        )

        # 3. Compilar
        _update_job(job_id, phase="Compilando el modelo", progress=8)
        model.compile(
            optimizer=_make_optimizer(request.optimizer, request.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # 4. Entrenar
        _update_job(job_id, phase="Entrenando", progress=10)
        model.fit(
            x_train,
            y_train,
            epochs=request.epochs,
            batch_size=request.batch_size,
            validation_split=request.validation_split,
            verbose=0,
            callbacks=[_JobProgressCallback(job_id, request.epochs)],
        )

        # 5. Evaluar sobre el conjunto de test
        _update_job(job_id, phase="Evaluando", progress=95)
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

        # 6. Guardar el modelo + sidecar de metadatos
        _update_job(job_id, phase="Guardando el modelo", progress=98)
        model_name = getattr(request.model_spec, "name", None) or request.dataset + "_cnn"
        model_path = save_built_model(model, model_name)
        # El entrenamiento normaliza a [0,1]: lo registramos para la inyección de fallos.
        with open(f"{model_path}.meta.json", "w") as f:
            json.dump({"normalize": True, "dataset": request.dataset}, f, indent=2)

        results = {
            "model_path": model_path,
            "model_name": model_name,
            "dataset": request.dataset,
            "epochs": request.epochs,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "parameters": int(model.count_params()),
        }
        _update_job(
            job_id, status="done", progress=100, phase="Completado", results=results
        )
        print(f"✅ Entrenamiento {job_id} completado: {model_path} (acc={test_acc:.4f})")

    except Exception as e:
        print(f"❌ Error en entrenamiento {job_id}: {e}")
        _update_job(job_id, status="error", phase="Error", error=str(e))


def start_training_job(request) -> str:
    """Crear un job de entrenamiento y lanzarlo en segundo plano."""
    job_id = f"train_{int(time.time())}_{str(uuid.uuid4())[:8]}"
    with _jobs_lock:
        _training_jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "phase": "En cola",
            "history": [],
            "error": None,
            "results": None,
        }

    thread = threading.Thread(target=_run_training, args=(job_id, request), daemon=True)
    thread.start()
    return job_id
