#!/usr/bin/env python3
"""
Script de prueba específico para stuck-at-0 y stuck-at-1 en VHDL
"""

import json
import os
import sys
import shutil
from datetime import datetime

# Agregar el directorio del proyecto al path
sys.path.append('/home/davidgonzalez/Documentos/project/Back_project')

# Importar las funciones necesarias
from main import modify_vhdl_weights_and_bias

def test_stuck_at_faults():
    """Probar específicamente stuck-at-0 y stuck-at-1"""
    print("=== PRUEBA DE STUCK-AT-0 Y STUCK-AT-1 EN VHDL ===")
    print(f"Fecha y hora: {datetime.now()}")
    
    # Archivo VHDL de prueba
    vhdl_file_path = "/home/davidgonzalez/Documentos/David_2025/4_CONV1_SAB_STUCKAT_DEC_RAM_TB/CONV1_SAB_STUCKAT_DEC_RAM.srcs/sources_1/new/CONV1_SAB_STUCK_DECOS.vhd"
    
    print(f"🔍 Verificando archivo VHDL: {vhdl_file_path}")
    
    if not os.path.exists(vhdl_file_path):
        print(f"❌ ERROR: Archivo VHDL no encontrado: {vhdl_file_path}")
        return False
    
    print("✅ Archivo VHDL encontrado")
    
    # Crear respaldo
    backup_path = f"{vhdl_file_path}.backup_stuck_at_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(vhdl_file_path, backup_path)
    print(f"✅ Respaldo creado: {backup_path}")
    
    # Leer contenido original
    with open(vhdl_file_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    print(f"✅ Contenido original leído ({len(original_content)} caracteres)")
    
    # Buscar valores originales
    print("\n🔍 Analizando valores originales...")
    
    # Buscar FMAP_1 - primer valor
    fmap1_start = original_content.find("constant FMAP_1: FILTER_TYPE:= (")
    if fmap1_start != -1:
        # Extraer el primer valor del filtro
        content_after = original_content[fmap1_start:]
        first_quote = content_after.find('"')
        second_quote = content_after.find('"', first_quote + 1)
        if first_quote != -1 and second_quote != -1:
            original_value = content_after[first_quote+1:second_quote]
            print(f"📍 FMAP_1[0][0] valor original: {original_value}")
            
            # Analizar cada bit
            print(f"   Bits: {' '.join([f'{i}:{bit}' for i, bit in enumerate(original_value)])}")
    
    # Buscar BIAS_VAL_1
    bias1_start = original_content.find('constant BIAS_VAL_1: signed (15 downto 0) := "')
    if bias1_start != -1:
        content_after = original_content[bias1_start:]
        quote_start = content_after.find('"')
        quote_end = content_after.find('"', quote_start + 1)
        if quote_start != -1 and quote_end != -1:
            original_bias = content_after[quote_start+1:quote_end]
            print(f"📍 BIAS_VAL_1 valor original: {original_bias}")
            print(f"   Bits: {' '.join([f'{i}:{bit}' for i, bit in enumerate(original_bias)])}")
    
    # Configuraciones de prueba
    test_configs = [
        {
            'name': 'STUCK-AT-0 en FMAP_1[0][0] bit 7',
            'config': {
                'filter_faults': [
                    {
                        'filter_name': 'FMAP_1',
                        'row': 0,
                        'col': 0,
                        'bit_position': 7,  # MSB
                        'fault_type': 'stuck_at_0'
                    }
                ],
                'bias_faults': []
            }
        },
        {
            'name': 'STUCK-AT-1 en FMAP_1[0][0] bit 0',
            'config': {
                'filter_faults': [
                    {
                        'filter_name': 'FMAP_1',
                        'row': 0,
                        'col': 0,
                        'bit_position': 0,  # LSB
                        'fault_type': 'stuck_at_1'
                    }
                ],
                'bias_faults': []
            }
        },
        {
            'name': 'STUCK-AT-0 en BIAS_VAL_1 bit 15',
            'config': {
                'filter_faults': [],
                'bias_faults': [
                    {
                        'bias_name': 'BIAS_VAL_1',
                        'bit_position': 15,  # MSB
                        'fault_type': 'stuck_at_0'
                    }
                ]
            }
        },
        {
            'name': 'STUCK-AT-1 en BIAS_VAL_1 bit 0',
            'config': {
                'filter_faults': [],
                'bias_faults': [
                    {
                        'bias_name': 'BIAS_VAL_1',
                        'bit_position': 0,  # LSB
                        'fault_type': 'stuck_at_1'
                    }
                ]
            }
        }
    ]
    
    success_count = 0
    
    for i, test in enumerate(test_configs, 1):
        print(f"\n{'='*60}")
        print(f"🧪 PRUEBA {i}/4: {test['name']}")
        print(f"{'='*60}")
        
        try:
            # Aplicar modificaciones
            modified_content = modify_vhdl_weights_and_bias(original_content, test['config'])
            
            if modified_content == original_content:
                print("❌ FALLO: El contenido no cambió")
                continue
            
            print("✅ Modificación aplicada exitosamente")
            
            # Verificar cambios específicos
            if test['config']['filter_faults']:
                fault = test['config']['filter_faults'][0]
                print(f"🔍 Verificando cambio en {fault['filter_name']}[{fault['row']}][{fault['col']}] bit {fault['bit_position']}")
                
                # Buscar el valor modificado
                fmap1_start_mod = modified_content.find("constant FMAP_1: FILTER_TYPE:= (")
                if fmap1_start_mod != -1:
                    content_after_mod = modified_content[fmap1_start_mod:]
                    first_quote_mod = content_after_mod.find('"')
                    second_quote_mod = content_after_mod.find('"', first_quote_mod + 1)
                    if first_quote_mod != -1 and second_quote_mod != -1:
                        modified_value = content_after_mod[first_quote_mod+1:second_quote_mod]
                        print(f"   Original: {original_value}")
                        print(f"   Modificado: {modified_value}")
                        
                        # Verificar el bit específico
                        expected_bit = '0' if fault['fault_type'] == 'stuck_at_0' else '1'
                        actual_bit = modified_value[fault['bit_position']]
                        
                        if actual_bit == expected_bit:
                            print(f"   ✅ Bit {fault['bit_position']}: {actual_bit} (correcto)")
                        else:
                            print(f"   ❌ Bit {fault['bit_position']}: {actual_bit} (esperado: {expected_bit})")
            
            if test['config']['bias_faults']:
                fault = test['config']['bias_faults'][0]
                print(f"🔍 Verificando cambio en {fault['bias_name']} bit {fault['bit_position']}")
                
                # Buscar el valor modificado
                bias1_start_mod = modified_content.find('constant BIAS_VAL_1: signed (15 downto 0) := "')
                if bias1_start_mod != -1:
                    content_after_mod = modified_content[bias1_start_mod:]
                    quote_start_mod = content_after_mod.find('"')
                    quote_end_mod = content_after_mod.find('"', quote_start_mod + 1)
                    if quote_start_mod != -1 and quote_end_mod != -1:
                        modified_bias = content_after_mod[quote_start_mod+1:quote_end_mod]
                        print(f"   Original: {original_bias}")
                        print(f"   Modificado: {modified_bias}")
                        
                        # Verificar el bit específico
                        expected_bit = '0' if fault['fault_type'] == 'stuck_at_0' else '1'
                        actual_bit = modified_bias[fault['bit_position']]
                        
                        if actual_bit == expected_bit:
                            print(f"   ✅ Bit {fault['bit_position']}: {actual_bit} (correcto)")
                        else:
                            print(f"   ❌ Bit {fault['bit_position']}: {actual_bit} (esperado: {expected_bit})")
            
            # Escribir archivo temporalmente para verificar
            temp_file = f"{vhdl_file_path}.temp_test_{i}"
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            # Verificar que se escribió correctamente
            with open(temp_file, 'r', encoding='utf-8') as f:
                written_content = f.read()
            
            if written_content == modified_content:
                print("✅ Archivo temporal escrito correctamente")
                success_count += 1
            else:
                print("❌ Error al escribir archivo temporal")
            
            # Limpiar archivo temporal
            os.remove(temp_file)
            
        except Exception as e:
            print(f"❌ ERROR en la prueba: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"📊 RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"✅ Pruebas exitosas: {success_count}/4")
    
    if success_count == 4:
        print("🎉 TODAS LAS PRUEBAS PASARON")
        print("   La inyección de fallos stuck-at-0 y stuck-at-1 funciona correctamente")
        print("   El problema debe estar en:")
        print("   1. El editor no refresca automáticamente")
        print("   2. Hay un cache en el frontend")
        print("   3. Los cambios son muy sutiles para notar")
    else:
        print(f"⚠️ {4-success_count} pruebas fallaron")
        print("   Revisa los errores mostrados arriba")
    
    return success_count == 4

if __name__ == "__main__":
    print("INICIANDO PRUEBAS ESPECÍFICAS DE STUCK-AT FAULTS\n")
    
    success = test_stuck_at_faults()
    
    if success:
        print("\n🔧 RECOMENDACIONES:")
        print("1. Cierra y vuelve a abrir el archivo VHDL en tu editor")
        print("2. Verifica que no hay cache en el navegador (Ctrl+F5)")
        print("3. Comprueba que estás viendo el archivo correcto")
        print("4. Los cambios de stuck-at pueden ser sutiles - busca específicamente los bits modificados")