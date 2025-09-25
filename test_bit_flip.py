#!/usr/bin/env python3
"""
Script para probar la funcionalidad de bit flip y diagnosticar problemas.
"""

import struct
import numpy as np

def float32_to_bits(value: float) -> str:
    """Convertir float32 a representación binaria."""
    return format(struct.unpack('I', struct.pack('f', value))[0], '032b')

def bits_to_float32(bits: str) -> float:
    """Convertir representación binaria a float32."""
    return struct.unpack('f', struct.pack('I', int(bits, 2)))[0]

def inject_bitflip(value: float, bit_position: int) -> float:
    """Inyectar un bitflip en una posición específica."""
    if not (0 <= bit_position <= 31):
        return value
        
    bits = float32_to_bits(value)
    bit_list = list(bits)
    bit_list[bit_position] = '1' if bit_list[bit_position] == '0' else '0'
    modified_bits = ''.join(bit_list)
    
    return bits_to_float32(modified_bits)

def test_sign_bit_flip():
    """Probar el flip del bit de signo (bit 0 en representación big-endian)."""
    print("=== PRUEBA DEL BIT DE SIGNO ===")
    
    test_values = [0.203125, -0.875000, 1.0, -1.0, 0.5, -0.5]
    
    for value in test_values:
        bits_original = float32_to_bits(value)
        
        # Flip del bit de signo (bit 0)
        modified_value = inject_bitflip(value, 0)
        bits_modified = float32_to_bits(modified_value)
        
        print(f"Valor original: {value:10.6f}")
        print(f"Bits original: {bits_original}")
        print(f"Valor modificado: {modified_value:10.6f}")
        print(f"Bits modificado: {bits_modified}")
        print(f"Cambio de signo: {value < 0} → {modified_value < 0}")
        print("-" * 50)

def test_multiple_bit_flip():
    """Probar el flip de múltiples bits en el mismo valor."""
    print("\n=== PRUEBA DE MÚLTIPLES BITS ===")
    
    original_value = 0.203125
    print(f"Valor original: {original_value}")
    print(f"Bits original: {float32_to_bits(original_value)}")
    
    # Simular el comportamiento actual (problemático)
    print("\n--- Comportamiento actual (problemático) ---")
    current_value = original_value
    bit_positions = [15, 20, 25, 31]  # Bits comunes para probar
    
    for bit_pos in bit_positions:
        old_value = current_value
        current_value = inject_bitflip(current_value, bit_pos)
        print(f"Bit {bit_pos:2d}: {old_value:10.6f} → {current_value:10.6f}")
        print(f"         Bits: {float32_to_bits(current_value)}")
    
    # Comportamiento correcto (todos los bits desde el original)
    print("\n--- Comportamiento correcto ---")
    corrected_value = original_value
    bits = float32_to_bits(original_value)
    bit_list = list(bits)
    
    for bit_pos in bit_positions:
        bit_list[bit_pos] = '1' if bit_list[bit_pos] == '0' else '0'
        print(f"Flipping bit {bit_pos}: {bit_list[bit_pos]}")
    
    corrected_bits = ''.join(bit_list)
    corrected_value = bits_to_float32(corrected_bits)
    
    print(f"Valor final corregido: {corrected_value:10.6f}")
    print(f"Bits finales: {corrected_bits}")

def test_all_bits_flip():
    """Probar el flip de todos los bits."""
    print("\n=== PRUEBA DE TODOS LOS BITS ===")
    
    original_value = 0.203125
    print(f"Valor original: {original_value}")
    print(f"Bits original: {float32_to_bits(original_value)}")
    
    # Flip todos los bits
    bits = float32_to_bits(original_value)
    inverted_bits = ''.join('1' if bit == '0' else '0' for bit in bits)
    inverted_value = bits_to_float32(inverted_bits)
    
    print(f"Valor invertido: {inverted_value}")
    print(f"Bits invertidos: {inverted_bits}")

if __name__ == "__main__":
    test_sign_bit_flip()
    test_multiple_bit_flip()
    test_all_bits_flip()