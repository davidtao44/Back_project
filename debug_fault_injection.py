#!/usr/bin/env python3
"""
Script de debug para verificar la funcionalidad de inyección de fallos
"""

import numpy as np
import json
from fault_injection.bitflip_injector import BitflipFaultInjector

def test_fault_injector_directly():
    """Probar el inyector de fallos directamente"""
    print("=== PRUEBA DIRECTA DEL INYECTOR DE FALLOS ===")
    
    # Crear inyector
    injector = BitflipFaultInjector()
    
    # Configurar fallo para una capa
    fault_config = {
        'enabled': True,
        'num_faults': 5,
        'fault_type': 'random'
    }
    
    injector.configure_fault('test_layer', fault_config)
    
    # Crear activaciones de prueba
    activations = np.random.rand(10, 10, 32).astype(np.float32)
    print(f"Activaciones originales (primeros 5 valores): {activations.flat[:5]}")
    
    # Inyectar fallos
    modified_activations, injected_faults = injector.inject_faults_in_activations(
        activations, 'test_layer'
    )
    
    print(f"Número de fallos inyectados: {len(injected_faults)}")
    
    for i, fault in enumerate(injected_faults):
        print(f"Fallo {i+1}:")
        print(f"  Posición: {fault['position']}")
        print(f"  Bit: {fault['bit_position']}")
        print(f"  Original: {fault['original_value']:.6f}")
        print(f"  Modificado: {fault['modified_value']:.6f}")
        print(f"  Diferencia: {abs(fault['modified_value'] - fault['original_value']):.6f}")
    
    return len(injected_faults) > 0

def test_bit_positions():
    """Probar la distribución de posiciones de bits"""
    print("\n=== PRUEBA DE DISTRIBUCIÓN DE BITS ===")
    
    injector = BitflipFaultInjector()
    
    # Generar muchas posiciones de bits para ver la distribución
    bit_positions = []
    for _ in range(100):
        # Simular la lógica de generación de bits
        import random
        if random.random() < 0.9:
            bit_pos = random.randint(20, 30)  # Bits muy significativos
        else:
            bit_pos = random.randint(0, 19)   # Bits menos significativos
        bit_positions.append(bit_pos)
    
    # Contar distribución
    high_bits = sum(1 for bp in bit_positions if bp >= 20)
    low_bits = sum(1 for bp in bit_positions if bp < 20)
    
    print(f"Bits muy altos (20-30): {high_bits}/100 ({high_bits}%)")
    print(f"Bits bajos (0-19): {low_bits}/100 ({low_bits}%)")
    
    # Probar algunos bitflips específicos
    test_value = 1.5
    print(f"\nProbando bitflips en valor {test_value}:")
    
    for bit_pos in [31, 30, 25, 20, 15, 10, 5, 0]:
        modified = injector.inject_bitflip(test_value, bit_pos)
        diff = abs(modified - test_value)
        print(f"  Bit {bit_pos:2d}: {test_value:.6f} -> {modified:.6f} (diff: {diff:.6f})")

if __name__ == "__main__":
    print("INICIANDO DEBUG DE INYECCIÓN DE FALLOS\n")
    
    # Ejecutar pruebas
    test1_ok = test_fault_injector_directly()
    test_bit_positions()
    
    print(f"\n=== RESUMEN ===")
    print(f"Inyector directo: {'✅ OK' if test1_ok else '❌ FALLO'}")
    
    if test1_ok:
        print("\n✅ La inyección de fallos funciona correctamente.")
        print("El problema podría estar en:")
        print("1. La configuración que llega desde el frontend")
        print("2. Los fallos se inyectan en valores muy pequeños")
        print("3. Los fallos no afectan significativamente la predicción")
    else:
        print("\n❌ Hay problemas en la implementación de inyección de fallos.")