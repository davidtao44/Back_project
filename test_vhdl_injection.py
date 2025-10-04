#!/usr/bin/env python3
"""
Script de prueba para verificar la inyección de fallos en VHDL
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

def test_vhdl_fault_injection():
    """Probar la inyección de fallos en VHDL directamente"""
    print("=== PRUEBA DE INYECCIÓN DE FALLOS EN VHDL ===")
    print(f"Fecha y hora: {datetime.now()}")
    
    # Archivo VHDL de prueba
    vhdl_file_path = "/home/davidgonzalez/Documentos/David_2025/4_CONV1_SAB_STUCKAT_DEC_RAM_TB/CONV1_SAB_STUCKAT_DEC_RAM.srcs/sources_1/new/CONV1_SAB_STUCK_DECOS.vhd"
    
    print(f"🔍 Verificando archivo VHDL: {vhdl_file_path}")
    
    if not os.path.exists(vhdl_file_path):
        print(f"❌ ERROR: Archivo VHDL no encontrado: {vhdl_file_path}")
        return False
    
    print("✅ Archivo VHDL encontrado")
    
    # Crear respaldo
    backup_path = f"{vhdl_file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(vhdl_file_path, backup_path)
    print(f"✅ Respaldo creado: {backup_path}")
    
    # Leer contenido original
    with open(vhdl_file_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    print(f"✅ Contenido original leído ({len(original_content)} caracteres)")
    
    # Buscar valores originales para verificar cambios
    print("\n🔍 Buscando valores originales...")
    
    # Buscar FMAP_1
    fmap1_start = original_content.find("constant FMAP_1: FILTER_TYPE:= (")
    if fmap1_start != -1:
        fmap1_section = original_content[fmap1_start:fmap1_start+500]
        print("📍 FMAP_1 encontrado:")
        print(fmap1_section[:200] + "...")
    
    # Buscar BIAS_VAL_1
    bias1_start = original_content.find("constant BIAS_VAL_1:")
    if bias1_start != -1:
        bias1_section = original_content[bias1_start:bias1_start+100]
        print("📍 BIAS_VAL_1 encontrado:")
        print(bias1_section)
    
    # Configuración de fallos de prueba
    fault_config = {
        'filter_faults': [
            {
                'filter_name': 'FMAP_1',
                'row': 0,
                'col': 0,
                'bit_position': 0,
                'fault_type': 'bitflip'
            }
        ],
        'bias_faults': [
            {
                'bias_name': 'BIAS_VAL_1',
                'bit_position': 0,
                'fault_type': 'bitflip'
            }
        ]
    }
    
    print(f"\n🔧 Aplicando configuración de fallos:")
    print(f"   - Filter faults: {fault_config['filter_faults']}")
    print(f"   - Bias faults: {fault_config['bias_faults']}")
    
    try:
        # Aplicar modificaciones
        modified_content = modify_vhdl_weights_and_bias(original_content, fault_config)
        print("✅ Modificaciones aplicadas exitosamente")
        
        # Verificar si hay cambios
        if modified_content == original_content:
            print("⚠️ ADVERTENCIA: El contenido no cambió después de la modificación")
            return False
        else:
            print("✅ El contenido fue modificado")
            
            # Mostrar diferencias
            print("\n🔍 Verificando cambios...")
            
            # Buscar FMAP_1 modificado
            fmap1_start_mod = modified_content.find("constant FMAP_1: FILTER_TYPE:= (")
            if fmap1_start_mod != -1:
                fmap1_section_mod = modified_content[fmap1_start_mod:fmap1_start_mod+500]
                print("📍 FMAP_1 modificado:")
                print(fmap1_section_mod[:200] + "...")
            
            # Buscar BIAS_VAL_1 modificado
            bias1_start_mod = modified_content.find("constant BIAS_VAL_1:")
            if bias1_start_mod != -1:
                bias1_section_mod = modified_content[bias1_start_mod:bias1_start_mod+100]
                print("📍 BIAS_VAL_1 modificado:")
                print(bias1_section_mod)
        
        # Escribir archivo modificado
        with open(vhdl_file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        print("✅ Archivo VHDL modificado guardado")
        
        # Verificar que el archivo se escribió correctamente
        with open(vhdl_file_path, 'r', encoding='utf-8') as f:
            written_content = f.read()
        
        if written_content == modified_content:
            print("✅ Verificación: El archivo se escribió correctamente")
        else:
            print("❌ ERROR: El archivo no se escribió correctamente")
            return False
        
        print(f"\n📊 Estadísticas:")
        print(f"   - Tamaño original: {len(original_content)} caracteres")
        print(f"   - Tamaño modificado: {len(modified_content)} caracteres")
        print(f"   - Diferencia: {len(modified_content) - len(original_content)} caracteres")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR durante la modificación: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Restaurar archivo original
        print(f"\n🔄 Restaurando archivo original desde: {backup_path}")
        shutil.copy2(backup_path, vhdl_file_path)
        print("✅ Archivo original restaurado")

def check_file_permissions():
    """Verificar permisos del archivo VHDL"""
    vhdl_file_path = "/home/davidgonzalez/Documentos/David_2025/4_CONV1_SAB_STUCKAT_DEC_RAM_TB/CONV1_SAB_STUCKAT_DEC_RAM.srcs/sources_1/new/CONV1_SAB_STUCK_DECOS.vhd"
    
    print("\n=== VERIFICACIÓN DE PERMISOS ===")
    
    if os.path.exists(vhdl_file_path):
        stat_info = os.stat(vhdl_file_path)
        print(f"📁 Archivo: {vhdl_file_path}")
        print(f"📏 Tamaño: {stat_info.st_size} bytes")
        print(f"🔐 Permisos: {oct(stat_info.st_mode)[-3:]}")
        print(f"📅 Última modificación: {datetime.fromtimestamp(stat_info.st_mtime)}")
        
        # Verificar si es escribible
        if os.access(vhdl_file_path, os.W_OK):
            print("✅ El archivo es escribible")
        else:
            print("❌ El archivo NO es escribible")
            
        # Verificar directorio padre
        parent_dir = os.path.dirname(vhdl_file_path)
        if os.access(parent_dir, os.W_OK):
            print("✅ El directorio padre es escribible")
        else:
            print("❌ El directorio padre NO es escribible")
    else:
        print(f"❌ Archivo no encontrado: {vhdl_file_path}")

if __name__ == "__main__":
    print("INICIANDO PRUEBAS DE INYECCIÓN DE FALLOS VHDL\n")
    
    # Verificar permisos
    check_file_permissions()
    
    # Ejecutar prueba
    success = test_vhdl_fault_injection()
    
    print(f"\n=== RESUMEN ===")
    if success:
        print("✅ La inyección de fallos VHDL funciona correctamente")
        print("   El problema podría estar en:")
        print("   1. El editor no está refrescando el archivo")
        print("   2. Hay un problema en el frontend")
        print("   3. Los cambios se están aplicando pero son muy sutiles")
    else:
        print("❌ Hay problemas en la inyección de fallos VHDL")
        print("   Revisa los errores mostrados arriba")