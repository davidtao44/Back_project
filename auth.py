from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from firebase_config import get_firestore_client
# Eliminada importación de FieldFilter - usando sintaxis simple de Firestore

# Configuración de seguridad
SECRET_KEY = "tu_clave_secreta_super_segura_cambiala_en_produccion"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Configuración para hash de contraseñas
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Modelos Pydantic
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: dict

class TokenData(BaseModel):
    username: Optional[str] = None

# Configuración de Firestore
USERS_COLLECTION = "users"

# Función para inicializar usuarios por defecto
def initialize_default_users():
    """Inicializa usuarios por defecto en Firestore si no existen"""
    try:
        db = get_firestore_client()
        users_ref = db.collection(USERS_COLLECTION)
        
        # Usuario admin por defecto
        admin_doc = users_ref.document("admin").get()
        if not admin_doc.exists:
            admin_user = {
                "username": "admin",
                "email": "admin@example.com",
                "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            users_ref.document("admin").set(admin_user)
            print("✅ Usuario admin creado en Firestore")
        
        # Usuario de prueba por defecto
        test_doc = users_ref.document("testuser").get()
        if not test_doc.exists:
            test_user = {
                "username": "testuser",
                "email": "test@example.com",
                "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
                "is_active": True,
                "created_at": datetime.utcnow()
            }
            users_ref.document("testuser").set(test_user)
            print("✅ Usuario testuser creado en Firestore")
            
    except Exception as e:
        print(f"⚠️ Error al inicializar usuarios por defecto: {str(e)}")

# Funciones de utilidad
def verify_password(plain_password, hashed_password):
    """Verificar contraseña"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Generar hash de contraseña"""
    return pwd_context.hash(password)

def get_user(username: str):
    """Obtener usuario de Firestore"""
    try:
        db = get_firestore_client()
        users_ref = db.collection(USERS_COLLECTION)
        user_doc = users_ref.document(username).get()
        
        if user_doc.exists:
            return user_doc.to_dict()
        return None
    except Exception as e:
        print(f"Error al obtener usuario {username}: {str(e)}")
        return None

def authenticate_user(username: str, password: str):
    """Autenticar usuario"""
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Crear token JWT"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verificar token JWT"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudieron validar las credenciales",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Obtener usuario actual desde el token"""
    return verify_token(credentials)

def create_user(user_data: UserCreate):
    """Crear nuevo usuario en Firestore"""
    try:
        db = get_firestore_client()
        users_ref = db.collection(USERS_COLLECTION)
        
        # Verificar si el usuario ya existe
        existing_user = users_ref.document(user_data.username).get()
        if existing_user.exists:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El usuario ya existe"
            )
        
        # Verificar si el email ya existe
        email_query = users_ref.where("email", "==", user_data.email).limit(1).get()
        if len(email_query) > 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El email ya está registrado"
            )
        
        # Crear nuevo usuario
        hashed_password = get_password_hash(user_data.password)
        new_user = {
            "username": user_data.username,
            "email": user_data.email,
            "hashed_password": hashed_password,
            "is_active": True,
            "created_at": datetime.utcnow()
        }
        
        # Guardar en Firestore
        users_ref.document(user_data.username).set(new_user)
        
        return {
            "username": new_user["username"],
            "email": new_user["email"],
            "is_active": new_user["is_active"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error al crear usuario: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor al crear usuario"
        )