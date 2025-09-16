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
layer_outputs/
├── user_abc123_1703123456_a1b2c3d4/
│   ├── conv2d_1_feature_maps.xlsx
│   ├── conv2d_1_channel_0.png
│   ├── conv2d_1_channel_1.png
│   └── ...
├── user_def456_1703123789_e5f6g7h8/
│   ├── conv2d_1_feature_maps.xlsx
│   └── ...
└── user_ghi789_1703124012_i9j0k1l2/
    └── ...
```

### 3. Endpoints Disponibles

#### `/manual_inference` (POST)
- Realiza inferencia manual y crea carpeta de sesión única
- Retorna rutas de archivos relativas a la carpeta de sesión

#### `/download_file/` (GET)
- Descarga archivos buscando automáticamente en todas las carpetas de sesión
- Parámetro: `file_path` (nombre del archivo)

#### `/list_sessions/` (GET)
- Lista todas las sesiones activas con información:
  - Session ID
  - Fecha de creación
  - Edad en horas
  - Número de archivos

#### `/cleanup_sessions/` (DELETE)
- Limpia sesiones antiguas
- Parámetro opcional: `max_age_hours` (default: 24 horas)

## Ventajas del Sistema

1. **Aislamiento**: Cada usuario tiene sus propios archivos
2. **Concurrencia**: Múltiples usuarios pueden trabajar simultáneamente
3. **Trazabilidad**: Cada sesión está identificada con usuario y timestamp
4. **Limpieza Automática**: Sistema de limpieza de sesiones antiguas
5. **Compatibilidad**: El frontend no necesita cambios, funciona transparentemente

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