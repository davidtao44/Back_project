#!/usr/bin/env python3
"""
Script de prueba para verificar que los fallos determinísticos en pesos están funcionando correctamente.
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import io

# Agregar el directorio padre al path para importar módulos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fault_injection.manual_inference import ManualInference

def create_test_image():
    """Crear una imagen de prueba simple."""
    # Crear una imagen de 32x32 con un patrón simple
    image = np.zeros((32, 32), dtype=np.uint8)
    
    # Agregar un patrón reconocible (cuadrado en el centro)
    image[10:22, 10:22] = 255
    image[12:20, 12:20] = 128
    
    # Convertir a bytes
    pil_image = Image.fromarray(image, mode='L')
    img_bytes = io.BytesIO()
    pil_image.save(img_bytes, format='PNG')
    return img_bytes.getvalue()

def test_weight_fault_injection():
    """Probar la inyección de fallos en pesos."""
    
    print("🧪 INICIANDO PRUEBA DE FALLOS EN PESOS")
    print("=" * 60)
    
    # Configuración de fallos en pesos
    weight_fault_config = {
        'weight_faults': {
            'enabled': True,
            'layers': {
                'conv2d_1': {
                    'target_type': 'kernel',  # Modificar kernels de la primera capa convolucional
                    'positions': [(0, 0, 0, 0), (1, 1, 0, 1)],  # Dos posiciones específicas
                    'bit_positions': [15, 20],  # Bits a modificar
                    'fault_type': 'specific'
                },
                'conv2d_2': {
                    'target_type': 'bias',  # Modificar bias de la segunda capa convolucional
                    'positions': [(0,), (1,)],  # Dos bias
                    'bit_positions': [10, 25],
                    'fault_type': 'specific'
                }
            }
        }
    }
    
    # Ruta del modelo (ajustar según tu configuración)
    model_path = "models/lenet5_mnist.h5"  # Ajusta esta ruta
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado en: {model_path}")
        print("Por favor, ajusta la ruta del modelo en el script.")
        return False
    
    try:
        # Crear imagen de prueba
        test_image = create_test_image()
        
        # 1. INFERENCIA GOLDEN (sin fallos)
        print("\n1️⃣ EJECUTANDO INFERENCIA GOLDEN (sin fallos)")
        print("-" * 50)
        
        golden_inference = ManualInference(
            model_path=model_path,
            output_dir="test_outputs",
            session_id="golden_test",
            fault_config=None  # Sin fallos
        )
        
        golden_results = golden_inference.perform_manual_inference(test_image)
        print(f"✅ Inferencia golden completada")
        print(f"   Predicción final: {golden_results['final_prediction']}")
        
        # 2. INFERENCIA CON FALLOS EN PESOS
        print("\n2️⃣ EJECUTANDO INFERENCIA CON FALLOS EN PESOS")
        print("-" * 50)
        
        faulty_inference = ManualInference(
            model_path=model_path,
            output_dir="test_outputs",
            session_id="weight_fault_test",
            fault_config=weight_fault_config
        )
        
        # Diagnóstico de pesos ANTES de la inferencia
        print("\n🔍 DIAGNÓSTICO DE PESOS:")
        weight_diagnosis = faulty_inference.diagnose_weight_changes()
        
        print(f"Fallos en pesos habilitados: {weight_diagnosis['weight_fault_enabled']}")
        print(f"Diferencias encontradas: {weight_diagnosis['weight_differences_found']}")
        print(f"Total de pesos modificados: {weight_diagnosis['total_modified_weights']}")
        
        if weight_diagnosis['weight_differences_found']:
            print("\n✅ FALLOS EN PESOS DETECTADOS:")
            for layer_name, layer_info in weight_diagnosis['layer_details'].items():
                if layer_info['differences_found']:
                    print(f"  📍 {layer_name}: {layer_info['total_differences']} diferencias")
                    for tensor_info in layer_info['weight_tensors']:
                        if tensor_info['modified_elements'] > 0:
                            print(f"    - {tensor_info['tensor_type']}: {tensor_info['modified_elements']} elementos modificados")
                            print(f"      Diferencia máxima: {tensor_info['max_absolute_difference']:.8f}")
                            
                            # Mostrar algunas posiciones modificadas
                            for pos_info in tensor_info['modified_positions'][:3]:
                                print(f"      Pos {pos_info['position']}: {pos_info['original_value']:.6f} → {pos_info['modified_value']:.6f}")
        else:
            print("❌ NO SE DETECTARON FALLOS EN PESOS")
            print("Posibles causas:")
            print("  - La configuración de fallos no se aplicó correctamente")
            print("  - Las posiciones especificadas son inválidas")
            print("  - El modelo no tiene las capas esperadas")
            return False
        
        # Ejecutar inferencia con fallos
        faulty_results = faulty_inference.perform_manual_inference(test_image)
        print(f"\n✅ Inferencia con fallos completada")
        print(f"   Predicción final: {faulty_results['final_prediction']}")
        
        # 3. COMPARAR RESULTADOS
        print("\n3️⃣ COMPARANDO RESULTADOS")
        print("-" * 50)
        
        # Comparar predicciones finales
        golden_pred = golden_results['final_prediction']
        faulty_pred = faulty_results['final_prediction']
        
        print(f"Predicción Golden: {golden_pred}")
        print(f"Predicción con Fallos: {faulty_pred}")
        
        # Verificar si hay diferencias en las predicciones
        if 'probabilities' in golden_pred and 'probabilities' in faulty_pred:
            golden_probs = np.array(golden_pred['probabilities'])
            faulty_probs = np.array(faulty_pred['probabilities'])
            
            prob_diff = np.abs(golden_probs - faulty_probs)
            max_prob_diff = np.max(prob_diff)
            mean_prob_diff = np.mean(prob_diff)
            
            print(f"\nDiferencias en probabilidades:")
            print(f"  Diferencia máxima: {max_prob_diff:.8f}")
            print(f"  Diferencia promedio: {mean_prob_diff:.8f}")
            
            if max_prob_diff > 1e-6:
                print("✅ Se detectaron diferencias en las probabilidades finales")
                
                # Mostrar las diferencias más significativas
                significant_diffs = np.where(prob_diff > 1e-6)[0]
                if len(significant_diffs) > 0:
                    print("Clases con diferencias significativas:")
                    for idx in significant_diffs[:5]:  # Mostrar las primeras 5
                        print(f"  Clase {idx}: {golden_probs[idx]:.6f} → {faulty_probs[idx]:.6f} (diff: {prob_diff[idx]:.6f})")
            else:
                print("⚠️ No se detectaron diferencias significativas en las probabilidades finales")
                print("Esto podría indicar que los fallos en pesos no están afectando la salida final")
        
        # 4. GENERAR REPORTE DETALLADO
        print("\n4️⃣ GENERANDO REPORTE DETALLADO")
        print("-" * 50)
        
        fault_summary = faulty_inference.get_fault_summary()
        print(f"Resumen de fallos:")
        print(f"  Total de fallos: {fault_summary['total_faults']}")
        print(f"  Fallos en pesos: {fault_summary['weight_faults']['total_faults']}")
        
        if fault_summary['weight_faults']['total_faults'] > 0:
            print("Detalles de fallos en pesos:")
            for fault in fault_summary['weight_faults']['fault_details'][:5]:  # Mostrar los primeros 5
                print(f"  - {fault['layer_name']}.{fault['target_type']}[{fault['position']}] bit {fault['bit_position']}")
                print(f"    {fault['original_value']:.6f} → {fault['modified_value']:.6f}")
        
        print(f"\n✅ PRUEBA COMPLETADA")
        print(f"Archivos de salida en: test_outputs/")
        
        return True
        
    except Exception as e:
        print(f"❌ Error durante la prueba: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_weight_fault_injection()
    
    if success:
        print("\n🎉 La prueba se ejecutó correctamente")
        print("Revisa los archivos de salida para verificar las diferencias")
    else:
        print("\n💥 La prueba falló")
        print("Revisa la configuración y los mensajes de error")