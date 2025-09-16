# Gestión de Sesiones Múltiples

## Descripción

Este sistema permite manejar múltiples usuarios simultáneos creando carpetas de sesión únicas para cada inferencia manual. Cada usuario obtiene su propia carpeta aislada donde se almacenan sus archivos de resultados.

## Cómo Funciona

### 1. Generación de Session ID
Cada vez que un usuario realiza una inferencia manual, se genera un `session_id` único que incluye:
- ID del usuario (UID de Firebase o 'anonymous')
- Timestamp actual
- UUID corto de 8 caracteres

Formato: `user_{uid}_{timestamp}_{uuid}`
Ejemplo: `user_abc123_1703123456_a1b2c3d4`

### 2. Estructura de Carpetas
```
Back_project/
├── layer_outputs/                    # Salidas de inferencia manual
│   ├── user_abc123_1703123456_a1b2c3d4/
│   │   ├── conv2d_1_feature_maps.xlsx
│   │   ├── conv2d_1_channel_0.png
│   │   ├── conv2d_1_channel_1.png
│   │   └── ...
│   ├── user_def456_1703123789_e5f6g7h8/
│   │   ├── conv2d_1_feature_maps.xlsx
│   │   └── ...
│   └── user_ghi789_1703124012_i9j0k1l2/
│       └── ...
├── vhdl_outputs/                     # Archivos VHDL de conversión de imagen
│   ├── user_abc123_1703123456_a1b2c3d4/
│   │   └── Memoria_Imagen_user_abc123_1703123456_a1b2c3d4.vhdl.txt
│   └── ...
├── model_weights_outputs/            # Pesos y bias extraídos del modelo
│   ├── user_abc123_1703123456_a1b2c3d4/
│   │   ├── conv2d_weights.txt
│   │   ├── conv2d_bias.txt
│   │   └── ...
│   └── ...
└── main.py
```

### 3. Endpoints Disponibles

#### 1. Inferencia Manual
- **Endpoint**: `POST /manual_inference`
- **Función**: Ejecuta inferencia y genera archivos únicos por sesión
- **Carpeta**: `layer_outputs/session_id/`
- **Respuesta**: Incluye `session_id` para identificar los archivos generados

#### 2. Conversión de Imagen a VHDL
- **Endpoint**: `POST /convert_image_to_vhdl/`
- **Función**: Convierte una imagen a código VHDL con sesión única
- **Carpeta**: `vhdl_outputs/session_id/`
- **Respuesta**: Incluye `session_id` y nombre del archivo VHDL generado

#### 3. Extracción de Pesos del Modelo
- **Endpoint**: `POST /extract_model_weights/`
- **Función**: Extrae pesos y bias del modelo en formato VHDL con sesión única
- **Carpeta**: `model_weights_outputs/session_id/`
- **Respuesta**: Incluye `session_id` y lista de archivos generados

#### 4. Descarga de Archivos
- **Endpoint**: `GET /download_file/{file_path}`
- **Función**: Descarga archivos buscando automáticamente en todas las carpetas de sesión
- **Búsqueda inteligente**: Busca en `layer_outputs`, `vhdl_outputs`, `model_weights_outputs`

#### 5. Limpieza de Sesiones
- **Endpoint**: `DELETE /cleanup_sessions/?max_age_hours=24`
- **Función**: Elimina carpetas de sesión más antiguas que el tiempo especificado
- **Alcance**: Limpia todas las carpetas de sesión (layer_outputs, vhdl_outputs, model_weights_outputs)
- **Parámetro**: `max_age_hours` (por defecto: 24 horas)

#### 6. Listado de Sesiones
- **Endpoint**: `GET /list_sessions/`
- **Función**: Lista todas las sesiones activas con información detallada
- **Información**: ID de sesión, tipo de sesión, tiempo de creación, edad, cantidad de archivos

## Ventajas del Sistema

1. **Aislamiento**: Cada usuario tiene sus propios archivos
2. **Concurrencia**: Múltiples usuarios pueden trabajar simultáneamente
3. **Trazabilidad**: Cada sesión está identificada con usuario y timestamp
4. **Limpieza Automática**: Sistema de limpieza de sesiones antiguas
5. **Compatibilidad**: El frontend no necesita cambios, funciona transparentemente

## Ejemplos de Uso

### Flujo Completo de Trabajo

```bash
# 1. Realizar inferencia manual
curl -X POST "http://localhost:8003/manual_inference" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"model_path": "path/to/model", "input_data": "data"}'

# 2. Convertir imagen a VHDL
curl -X POST "http://localhost:8003/convert_image_to_vhdl/" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -F "image=@image.jpg"

# 3. Extraer pesos del modelo
curl -X POST "http://localhost:8003/extract_model_weights/" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"model_path": "path/to/model.h5"}'

# 4. Descargar archivo generado
curl -X GET "http://localhost:8003/download_file/archivo.txt" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -o "archivo_descargado.txt"
```

### Gestión de Sesiones

```bash
# Listar todas las sesiones activas
curl -X GET "http://localhost:8003/list_sessions/" \
     -H "Authorization: Bearer YOUR_TOKEN"

# Limpiar sesiones más antiguas que 24 horas (por defecto)
curl -X DELETE "http://localhost:8003/cleanup_sessions/" \
     -H "Authorization: Bearer YOUR_TOKEN"

# Limpiar sesiones más antiguas que 1 hora
curl -X DELETE "http://localhost:8003/cleanup_sessions/?max_age_hours=1" \
     -H "Authorization: Bearer YOUR_TOKEN"

# Limpiar sesiones más antiguas que 30 minutos
curl -X DELETE "http://localhost:8003/cleanup_sessions/?max_age_hours=0.5" \
     -H "Authorization: Bearer YOUR_TOKEN"
```

## Mantenimiento

### Limpieza Manual
```bash
# Eliminar sesiones más antiguas que 12 horas
curl -X DELETE "http://localhost:8003/cleanup_sessions/?max_age_hours=12" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Monitoreo
```bash
# Listar sesiones activas
curl -X GET "http://localhost:8003/list_sessions/" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Consideraciones de Rendimiento

- Las carpetas de sesión se crean automáticamente
- Los archivos se buscan eficientemente en todas las sesiones
- Se recomienda ejecutar limpieza periódica para evitar acumulación de archivos
- El sistema es escalable para decenas de usuarios simultáneos

## Seguridad

- Cada usuario solo puede acceder a sus propios archivos a través de la autenticación
- Los session IDs incluyen el UID del usuario para trazabilidad
- Los archivos se almacenan en carpetas separadas por sesión