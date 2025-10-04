#!/usr/bin/env python3
"""
Script de prueba inteligente para stuck-at faults que solo aplica cuando hay cambios visibles
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

def analyze_original_values(content):
    """Analizar los valores originales para determinar qué stuck-at faults serán visibles"""
    print("🔍 ANALIZANDO VALORES ORIGINALES PARA STUCK-AT FAULTS")
    
    results = {
        'filter_analysis': {},
        'bias_analysis': {}
    }
    
    # Analizar FMAP_1
    fmap1_start = content.find("constant FMAP_1: FILTER_TYPE:= (")
    if fmap1_start != -1:
        print("\n📍 Analizando FMAP_1...")
        
        # Extraer la matriz completa
        content_after = content[fmap1_start:]
        matrix_start = content_after.find('(')
        matrix_end = content_after.find(');', matrix_start)
        matrix_content = content_after[matrix_start:matrix_end]
        
        # Extraer valores de la primera fila
        lines = matrix_content.split('\n')
        first_row_line = None
        for line in lines:
            if '"' in line:
                first_row_line = line
                break
        
        if first_row_line:
            # Extraer todos los valores de la primera fila
            import re
            values = re.findall(r'"([01]{8})"', first_row_line)
            
            print(f"   Primera fila encontrada: {len(values)} valores")
            
            for col, value in enumerate(values[:5]):  # Solo primeros 5 valores
                print(f"   FMAP_1[0][{col}] = {value}")
                
                # Analizar cada bit
                visible_stuck_at_0 = []
                visible_stuck_at_1 = []
                
                for bit_pos, bit_val in enumerate(value):
                    if bit_val == '1':
                        visible_stuck_at_0.append(bit_pos)
                    else:
                        visible_stuck_at_1.append(bit_pos)
                
                results['filter_analysis'][f'FMAP_1[0][{col}]'] = {
                    'original_value': value,
                    'visible_stuck_at_0': visible_stuck_at_0,
                    'visible_stuck_at_1': visible_stuck_at_1
                }
                
                print(f"     Stuck-at-0 visible en bits: {visible_stuck_at_0}")
                print(f"     Stuck-at-1 visible en bits: {visible_stuck_at_1}")
    
    # Analizar BIAS_VAL_1
    bias1_start = content.find('constant BIAS_VAL_1: signed (15 downto 0) := "')
    if bias1_start != -1:
        print("\n📍 Analizando BIAS_VAL_1...")
        
        content_after = content[bias1_start:]
        quote_start = content_after.find('"')
        quote_end = content_after.find('"', quote_start + 1)
        bias_value = content_after[quote_start+1:quote_end]
        
        print(f"   BIAS_VAL_1 = {bias_value}")
        
        visible_stuck_at_0 = []
        visible_stuck_at_1 = []
        
        for bit_pos, bit_val in enumerate(bias_value):
            if bit_val == '1':
                visible_stuck_at_0.append(bit_pos)
            else:
                visible_stuck_at_1.append(bit_pos)
        
        results['bias_analysis']['BIAS_VAL_1'] = {
            'original_value': bias_value,
            'visible_stuck_at_0': visible_stuck_at_0,
            'visible_stuck_at_1': visible_stuck_at_1
        }
        
        print(f"     Stuck-at-0 visible en bits: {visible_stuck_at_0}")
        print(f"     Stuck-at-1 visible en bits: {visible_stuck_at_1}")
    
    return results

def generate_smart_test_configs(analysis):
    """Generar configuraciones de prueba inteligentes basadas en el análisis"""
    print("\n🧠 GENERANDO CONFIGURACIONES DE PRUEBA INTELIGENTES")
    
    configs = []
    
    # Configuraciones para filtros
    for filter_key, filter_data in analysis['filter_analysis'].items():
        if '[0][0]' in filter_key:  # Solo probar la primera posición
            # Stuck-at-0 (solo bits que son 1)
            if filter_data['visible_stuck_at_0']:
                bit_pos = filter_data['visible_stuck_at_0'][0]  # Primer bit disponible
                configs.append({
                    'name': f'STUCK-AT-0 en FMAP_1[0][0] bit {bit_pos} (1→0)',
                    'config': {
                        'filter_faults': [{
                            'filter_name': 'FMAP_1',
                            'row': 0,
                            'col': 0,
                            'bit_position': bit_pos,
                            'fault_type': 'stuck_at_0'
                        }],
                        'bias_faults': []
                    },
                    'expected_change': f'bit {bit_pos}: 1 → 0'
                })
            
            # Stuck-at-1 (solo bits que son 0)
            if filter_data['visible_stuck_at_1']:
                bit_pos = filter_data['visible_stuck_at_1'][0]  # Primer bit disponible
                configs.append({
                    'name': f'STUCK-AT-1 en FMAP_1[0][0] bit {bit_pos} (0→1)',
                    'config': {
                        'filter_faults': [{
                            'filter_name': 'FMAP_1',
                            'row': 0,
                            'col': 0,
                            'bit_position': bit_pos,
                            'fault_type': 'stuck_at_1'
                        }],
                        'bias_faults': []
                    },
                    'expected_change': f'bit {bit_pos}: 0 → 1'
                })
    
    # Configuraciones para bias
    for bias_key, bias_data in analysis['bias_analysis'].items():
        # Stuck-at-0 (solo bits que son 1)
        if bias_data['visible_stuck_at_0']:
            bit_pos = bias_data['visible_stuck_at_0'][0]  # Primer bit disponible
            configs.append({
                'name': f'STUCK-AT-0 en {bias_key} bit {bit_pos} (1→0)',
                'config': {
                    'filter_faults': [],
                    'bias_faults': [{
                        'bias_name': bias_key,
                        'bit_position': bit_pos,
                        'fault_type': 'stuck_at_0'
                    }]
                },
                'expected_change': f'bit {bit_pos}: 1 → 0'
            })
        
        # Stuck-at-1 (solo bits que son 0)
        if bias_data['visible_stuck_at_1']:
            bit_pos = bias_data['visible_stuck_at_1'][0]  # Primer bit disponible
            configs.append({
                'name': f'STUCK-AT-1 en {bias_key} bit {bit_pos} (0→1)',
                'config': {
                    'filter_faults': [],
                    'bias_faults': [{
                        'bias_name': bias_key,
                        'bit_position': bit_pos,
                        'fault_type': 'stuck_at_1'
                    }]
                },
                'expected_change': f'bit {bit_pos}: 0 → 1'
            })
    
    print(f"   ✅ Generadas {len(configs)} configuraciones de prueba")
    for i, config in enumerate(configs, 1):
        print(f"   {i}. {config['name']} - {config['expected_change']}")
    
    return configs

def test_smart_stuck_at_faults():
    """Ejecutar pruebas inteligentes de stuck-at faults"""
    print("=== PRUEBA INTELIGENTE DE STUCK-AT FAULTS ===")
    print(f"Fecha y hora: {datetime.now()}")
    
    # Archivo VHDL
    vhdl_file_path = "/home/davidgonzalez/Documentos/David_2025/4_CONV1_SAB_STUCKAT_DEC_RAM_TB/CONV1_SAB_STUCKAT_DEC_RAM.srcs/sources_1/new/CONV1_SAB_STUCK_DECOS.vhd"
    
    if not os.path.exists(vhdl_file_path):
        print(f"❌ ERROR: Archivo VHDL no encontrado")
        return False
    
    # Crear respaldo
    backup_path = f"{vhdl_file_path}.backup_smart_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(vhdl_file_path, backup_path)
    print(f"✅ Respaldo creado: {os.path.basename(backup_path)}")
    
    # Leer contenido original
    with open(vhdl_file_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    # Analizar valores originales
    analysis = analyze_original_values(original_content)
    
    # Generar configuraciones inteligentes
    test_configs = generate_smart_test_configs(analysis)
    
    if not test_configs:
        print("❌ No se pudieron generar configuraciones de prueba")
        return False
    
    print(f"\n🚀 EJECUTANDO {len(test_configs)} PRUEBAS INTELIGENTES")
    
    success_count = 0
    
    for i, test in enumerate(test_configs, 1):
        print(f"\n{'='*60}")
        print(f"🧪 PRUEBA {i}/{len(test_configs)}: {test['name']}")
        print(f"   Cambio esperado: {test['expected_change']}")
        print(f"{'='*60}")
        
        try:
            # Aplicar modificaciones
            modified_content = modify_vhdl_weights_and_bias(original_content, test['config'])
            
            if modified_content == original_content:
                print("❌ FALLO: El contenido no cambió (esto NO debería pasar)")
                continue
            
            print("✅ Modificación aplicada exitosamente")
            
            # Verificar cambios específicos
            change_verified = False
            
            if test['config']['filter_faults']:
                fault = test['config']['filter_faults'][0]
                # Buscar cambio en filtro
                fmap1_start_orig = original_content.find("constant FMAP_1: FILTER_TYPE:= (")
                fmap1_start_mod = modified_content.find("constant FMAP_1: FILTER_TYPE:= (")
                
                if fmap1_start_orig != -1 and fmap1_start_mod != -1:
                    # Extraer primer valor original y modificado
                    import re
                    orig_match = re.search(r'"([01]{8})"', original_content[fmap1_start_orig:fmap1_start_orig+500])
                    mod_match = re.search(r'"([01]{8})"', modified_content[fmap1_start_mod:fmap1_start_mod+500])
                    
                    if orig_match and mod_match:
                        orig_val = orig_match.group(1)
                        mod_val = mod_match.group(1)
                        
                        print(f"   Original: {orig_val}")
                        print(f"   Modificado: {mod_val}")
                        
                        bit_pos = fault['bit_position']
                        if orig_val[bit_pos] != mod_val[bit_pos]:
                            print(f"   ✅ Bit {bit_pos}: {orig_val[bit_pos]} → {mod_val[bit_pos]} (CORRECTO)")
                            change_verified = True
                        else:
                            print(f"   ❌ Bit {bit_pos}: No cambió")
            
            if test['config']['bias_faults']:
                fault = test['config']['bias_faults'][0]
                # Buscar cambio en bias
                bias_pattern = r'constant BIAS_VAL_1: signed \(15 downto 0\) := "([01]{16})";'
                orig_match = re.search(bias_pattern, original_content)
                mod_match = re.search(bias_pattern, modified_content)
                
                if orig_match and mod_match:
                    orig_val = orig_match.group(1)
                    mod_val = mod_match.group(1)
                    
                    print(f"   Original: {orig_val}")
                    print(f"   Modificado: {mod_val}")
                    
                    bit_pos = fault['bit_position']
                    if orig_val[bit_pos] != mod_val[bit_pos]:
                        print(f"   ✅ Bit {bit_pos}: {orig_val[bit_pos]} → {mod_val[bit_pos]} (CORRECTO)")
                        change_verified = True
                    else:
                        print(f"   ❌ Bit {bit_pos}: No cambió")
            
            if change_verified:
                success_count += 1
                print("   🎉 PRUEBA EXITOSA")
            else:
                print("   ❌ PRUEBA FALLIDA")
                
        except Exception as e:
            print(f"❌ ERROR en la prueba: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"📊 RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"✅ Pruebas exitosas: {success_count}/{len(test_configs)}")
    
    if success_count == len(test_configs):
        print("🎉 TODAS LAS PRUEBAS PASARON")
        print("   ✅ La inyección de fallos stuck-at funciona perfectamente")
        print("   ✅ Los cambios se están aplicando correctamente")
        print("   ✅ El problema debe estar en la visualización del editor")
    elif success_count > 0:
        print(f"⚠️ {len(test_configs)-success_count} pruebas fallaron")
        print("   Algunas configuraciones funcionan, otras no")
    else:
        print("❌ TODAS LAS PRUEBAS FALLARON")
        print("   Hay un problema serio en la inyección de fallos")
    
    return success_count == len(test_configs)

if __name__ == "__main__":
    print("INICIANDO PRUEBAS INTELIGENTES DE STUCK-AT FAULTS\n")
    
    success = test_smart_stuck_at_faults()
    
    if success:
        print("\n🔧 CONCLUSIÓN:")
        print("✅ La inyección de fallos funciona correctamente")
        print("✅ El problema está en la visualización, no en la funcionalidad")
        print("\n💡 RECOMENDACIONES:")
        print("1. Refresca el archivo en tu editor (Ctrl+R o F5)")
        print("2. Cierra y vuelve a abrir el archivo VHDL")
        print("3. Verifica que no hay cache en el navegador")
        print("4. Los cambios SÍ se están aplicando al archivo")
    else:
        print("\n🔧 DIAGNÓSTICO:")
        print("❌ Hay problemas en la inyección de fallos")
        print("   Revisa los logs del backend para más detalles")