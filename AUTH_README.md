# Sistema de Autenticación - Backend FastAPI

Este documento explica cómo usar el sistema de autenticación JWT implementado en el backend.

## 🔐 Características

- **JWT (JSON Web Tokens)** para autenticación
- **Hashing de contraseñas** con bcrypt
- **Registro y login** de usuarios
- **Verificación de tokens**
- **Protección de endpoints** sensibles
- **Expiración automática** de tokens (30 minutos por defecto)

## 📋 Endpoints Disponibles

### Autenticación

#### 1. Registro de Usuario
```http
POST /auth/register
Content-Type: application/json

{
  "username": "usuario123",
  "email": "usuario@email.com",
  "password": "contraseña_segura"
}
```

**Respuesta exitosa:**
```json
{
  "success": true,
  "message": "Usuario creado exitosamente",
  "user": {
    "username": "usuario123",
    "email": "usuario@email.com",
    "is_active": true
  }
}
```

#### 2. Iniciar Sesión
```http
POST /auth/login
Content-Type: application/json

{
  "username": "usuario123",
  "password": "contraseña_segura"
}
```

**Respuesta exitosa:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user": {
    "username": "usuario123",
    "email": "usuario@email.com",
    "is_active": true
  }
}
```

#### 3. Obtener Información del Usuario Actual
```http
GET /auth/me
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

#### 4. Verificar Token
```http
POST /auth/verify-token
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## 🔒 Endpoints Protegidos

Los siguientes endpoints requieren autenticación:

- `POST /create_cnn/` - Crear modelo CNN
- `POST /delete_models/` - Eliminar modelos
- `POST /quantize_model/` - Cuantizar modelo

### Cómo usar endpoints protegidos:

```http
POST /create_cnn/
Authorization: Bearer tu_token_aqui
Content-Type: application/json

{
  "input_shape": [28, 28, 1],
  "num_classes": 10,
  "conv_layers": [
    {"filters": 32, "kernel_size": 3, "activation": "relu"},
    {"filters": 64, "kernel_size": 3, "activation": "relu"}
  ],
  "dense_layers": [
    {"units": 128, "activation": "relu"},
    {"units": 10, "activation": "softmax"}
  ]
}
```

## 🚀 Integración con Frontend React

Ve el archivo `frontend_auth_example.js` para un ejemplo completo de cómo integrar la autenticación en React.

### Pasos básicos:

1. **Instalar dependencias** (si usas fetch nativo, no necesitas nada extra)

2. **Modificar tu AuthContext** usando el ejemplo proporcionado

3. **Guardar el token** en localStorage después del login:
```javascript
const { login } = useAuth();

try {
  const result = await login('usuario123', 'contraseña');
  // Token se guarda automáticamente
  console.log('Login exitoso:', result);
} catch (error) {
  console.error('Error:', error.message);
}
```

4. **Hacer requests autenticados**:
```javascript
const { authenticatedFetch } = useAuth();

const crearModelo = async (config) => {
  try {
    const response = await authenticatedFetch('/create_cnn/', {
      method: 'POST',
      body: JSON.stringify(config)
    });
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error creando modelo:', error);
  }
};
```

## 🛠️ Configuración

### Variables de Entorno (Opcional)

Puedes configurar estas variables en un archivo `.env`:

```env
# Clave secreta para JWT (cambiar en producción)
SECRET_KEY=tu_clave_secreta_muy_segura_aqui

# Algoritmo de encriptación
ALGORITHM=HS256

# Tiempo de expiración del token en minutos
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### Base de Datos

Actualmente usa una "base de datos" simulada en memoria. Para producción, considera:

- **SQLite** para aplicaciones pequeñas
- **PostgreSQL** para aplicaciones grandes
- **MongoDB** para datos no relacionales

## 🧪 Pruebas

### Con curl:

```bash
# 1. Registrar usuario
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"username":"test","email":"test@test.com","password":"test123"}'

# 2. Hacer login
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"test123"}'

# 3. Usar el token (reemplazar TOKEN con el token recibido)
curl -X GET "http://localhost:8000/auth/me" \
  -H "Authorization: Bearer TOKEN"
```

### Con la documentación interactiva:

1. Ejecuta el servidor: `python main.py`
2. Ve a: `http://localhost:8000/docs`
3. Usa la interfaz de Swagger para probar los endpoints

## 🔧 Usuarios de Prueba

El sistema incluye usuarios de prueba:

```
Usuario: admin
Contraseña: admin123
Email: admin@example.com

Usuario: user
Contraseña: user123
Email: user@example.com
```

## ⚠️ Consideraciones de Seguridad

1. **Cambiar SECRET_KEY** en producción
2. **Usar HTTPS** en producción
3. **Configurar CORS** apropiadamente
4. **Implementar rate limiting** para prevenir ataques de fuerza bruta
5. **Usar una base de datos real** en lugar de la simulada
6. **Implementar refresh tokens** para mayor seguridad

## 🐛 Solución de Problemas

### Error 401 - Unauthorized
- Verifica que el token esté incluido en el header `Authorization`
- Asegúrate de usar el formato: `Bearer tu_token_aqui`
- El token puede haber expirado (30 minutos por defecto)

### Error 422 - Validation Error
- Verifica que todos los campos requeridos estén presentes
- Revisa el formato de los datos enviados

### Error 500 - Internal Server Error
- Revisa los logs del servidor
- Verifica que todas las dependencias estén instaladas

## 📚 Próximos Pasos

1. **Implementar base de datos real**
2. **Agregar roles y permisos**
3. **Implementar refresh tokens**
4. **Agregar recuperación de contraseña**
5. **Implementar rate limiting**
6. **Agregar logging de seguridad**