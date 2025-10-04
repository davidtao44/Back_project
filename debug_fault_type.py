#!/usr/bin/env python3
"""
Script de debug para verificar el comportamiento del tipo de fallo stuck_at_0
"""

def apply_bit_faults(binary_string: str, bit_positions: list, fault_type: str = 'bitflip') -> str:
    """
    Aplica fallos de bit según el tipo especificado a una cadena binaria.
    
    Args:
        binary_string: Cadena binaria original
        bit_positions: Lista de posiciones de bits a modificar
        fault_type: Tipo de fallo ('bitflip', 'stuck_at_0', 'stuck_at_1')
    
    Returns:
        Cadena binaria modificada
    """
    if not bit_positions:
        return binary_string
    
    # Convertir a lista para poder modificar
    bits = list(binary_string)
    
    for bit_pos in bit_positions:
        if 0 <= bit_pos < len(bits):
            original_bit = bits[bit_pos]
            
            if fault_type == 'stuck_at_0':
                # Forzar el bit a 0
                bits[bit_pos] = '0'
                print(f"  🔒 Bit {bit_pos}: {original_bit} → 0 (stuck-at-0)")
            elif fault_type == 'stuck_at_1':
                # Forzar el bit a 1
                bits[bit_pos] = '1'
                print(f"  🔒 Bit {bit_pos}: {original_bit} → 1 (stuck-at-1)")
            else:  # bitflip o cualquier otro tipo
                # Invertir el bit (comportamiento original)
                bits[bit_pos] = '1' if bits[bit_pos] == '0' else '0'
                print(f"  🔄 Bit {bit_pos}: {original_bit} → {bits[bit_pos]} (bit-flip)")
        else:
            print(f"  ⚠️ Posición de bit inválida: {bit_pos} (longitud: {len(bits)})")
    
    return ''.join(bits)

def test_fault_types():
    """Probar diferentes tipos de fallo"""
    print("=== PRUEBA DE TIPOS DE FALLO ===")
    
    # Valor binario de prueba (8 bits)
    original_value = "11110000"
    bit_position = 0  # Primer bit (que es 1)
    
    print(f"Valor original: {original_value}")
    print(f"Bit a modificar: posición {bit_position} (valor actual: {original_value[bit_position]})")
    print()
    
    # Probar stuck_at_0
    print("1. Probando stuck_at_0:")
    result_stuck_0 = apply_bit_faults(original_value, [bit_position], 'stuck_at_0')
    print(f"   Resultado: {result_stuck_0}")
    print()
    
    # Probar stuck_at_1
    print("2. Probando stuck_at_1:")
    result_stuck_1 = apply_bit_faults(original_value, [bit_position], 'stuck_at_1')
    print(f"   Resultado: {result_stuck_1}")
    print()
    
    # Probar bitflip
    print("3. Probando bitflip:")
    result_bitflip = apply_bit_faults(original_value, [bit_position], 'bitflip')
    print(f"   Resultado: {result_bitflip}")
    print()
    
    # Probar valor desconocido (debería hacer bitflip)
    print("4. Probando valor desconocido:")
    result_unknown = apply_bit_faults(original_value, [bit_position], 'unknown_type')
    print(f"   Resultado: {result_unknown}")
    print()

if __name__ == "__main__":
    test_fault_types()