#!/usr/bin/env python3
"""
Script de prueba para verificar la funcionalidad de inyección de fallos.
Demuestra cómo funciona el bitflip en diferentes posiciones de bits.
"""

import numpy as np
import struct
from fault_injection.bitflip_injector import BitflipFaultInjector

def float32_to_bits_detailed(value: float) -> str:
    """Convertir un float32 a su representación binaria con detalles."""
    packed = struct.pack('!f', value)
    bits = ''.join(format(byte, '08b') for byte in packed)
    
    # Separar en componentes IEEE 754
    sign_bit = bits[0]
    exponent = bits[1:9]
    mantissa = bits[9:32]
    
    print(f"Valor: {value}")
    print(f"Bits completos: {bits}")
    print(f"Signo (bit 31): {sign_bit}")
    print(f"Exponente (bits 30-23): {exponent}")
    print(f"Mantisa (bits 22-0): {mantissa}")
    print(f"Posiciones: {''.join([str(i%10) for i in range(32)])}")
    print(f"           {''.join([str(i//10) for i in range(32)])}")
    print()
    
    return bits

def test_bitflip_positions():
    """Probar inyección de fallos en diferentes posiciones de bits."""
    print("=== PRUEBA DE INYECCIÓN DE FALLOS ===\n")
    
    injector = BitflipFaultInjector()
    
    # Valor de prueba
    test_value = 0.5
    print(f"VALOR ORIGINAL: {test_value}")
    float32_to_bits_detailed(test_value)
    
    print("PROBANDO DIFERENTES POSICIONES DE BITS:")
    print("-" * 50)
    
    # Probar diferentes posiciones
    test_positions = [0, 1, 7, 8, 15, 16, 22, 23, 30, 31]
    
    for bit_pos in test_positions:
        try:
            modified_value = injector.inject_bitflip(test_value, bit_pos)
            difference = abs(modified_value - test_value)
            
            print(f"Bit {bit_pos:2d} (LSB=0): {test_value} -> {modified_value}")
            print(f"         Diferencia: {difference}")
            print(f"         Cambio significativo: {'SÍ' if difference > 0.001 else 'NO'}")
            print()
            
        except Exception as e:
            print(f"Error en posición {bit_pos}: {e}")

def test_fault_injection_on_array():
    """Probar inyección de fallos en un array de activaciones."""
    print("\n=== PRUEBA EN ARRAY DE ACTIVACIONES ===\n")
    
    injector = BitflipFaultInjector()
    
    # Crear array de prueba
    test_array = np.array([0.1, 0.5, 0.9, -0.3, 2.5], dtype=np.float32)
    print(f"Array original: {test_array}")
    
    # Configurar inyección de fallos
    injector.configure_fault('test_layer', {
        'enabled': True,
        'num_faults': 3,
        'fault_type': 'random'
    })
    
    # Aplicar fallos
    modified_array, fault_list = injector.inject_faults_in_activations(test_array, 'test_layer')
    
    print(f"Array modificado: {modified_array}")
    print(f"Diferencias: {modified_array - test_array}")
    
    print("\nDetalles de fallos inyectados:")
    for i, fault in enumerate(fault_list):
        print(f"Fallo {i+1}:")
        print(f"  Posición: {fault['position']}")
        print(f"  Bit: {fault['bit_position']}")
        print(f"  Valor original: {fault['original_value']}")
        print(f"  Valor modificado: {fault['modified_value']}")
        print(f"  Diferencia: {abs(fault['modified_value'] - fault['original_value'])}")
        print()

def test_specific_bit_ranges():
    """Probar el impacto de fallos en diferentes rangos de bits."""
    print("\n=== ANÁLISIS POR RANGOS DE BITS ===\n")
    
    injector = BitflipFaultInjector()
    test_value = 1.0
    
    print(f"Valor de prueba: {test_value}")
    
    ranges = {
        "LSB (0-7)": range(0, 8),
        "Medio (8-15)": range(8, 16),
        "Alto (16-22)": range(16, 23),
        "Exponente (23-30)": range(23, 31),
        "Signo (31)": [31]
    }
    
    for range_name, bit_range in ranges.items():
        print(f"\n{range_name}:")
        max_diff = 0
        min_diff = float('inf')
        
        for bit_pos in bit_range:
            modified = injector.inject_bitflip(test_value, bit_pos)
            diff = abs(modified - test_value)
            max_diff = max(max_diff, diff)
            min_diff = min(min_diff, diff) if diff > 0 else min_diff
            
            print(f"  Bit {bit_pos:2d}: {test_value} -> {modified:10.6f} (diff: {diff:10.6f})")
        
        print(f"  Rango de impacto: {min_diff:.6f} - {max_diff:.6f}")

if __name__ == "__main__":
    test_bitflip_positions()
    test_fault_injection_on_array()
    test_specific_bit_ranges()
    
    print("\n=== CONCLUSIONES ===")
    print("- Posición 0 = LSB (Least Significant Bit) - menor impacto")
    print("- Posición 31 = MSB (Most Significant Bit) - mayor impacto")
    print("- Bits 23-30 = Exponente - impacto muy alto")
    print("- Bits 0-22 = Mantisa - impacto variable")
    print("- Bit 31 = Signo - cambia el signo del número")