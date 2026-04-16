from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth
from jose import JWTError, jwt

from app.core.config import ACCESS_TOKEN_EXPIRE_MINUTES, ALGORITHM, SECRET_KEY, USERS_COLLECTION
from app.core.firebase import get_firestore_client
from app.schemas.auth import GoogleAuthRequest, GoogleTokenData

security = HTTPBearer()


def verify_google_token(id_token: str):
    """Verifica el token de Google usando Firebase Auth."""
    try:
        return auth.verify_id_token(id_token)
    except Exception as e:
        print(f"Error al verificar token de Google: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de Google inválido",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_or_create_user_from_google(decoded_token: dict):
    """Obtiene o crea un usuario en Firestore basado en la información de Google."""
    try:
        db = get_firestore_client()
        users_ref = db.collection(USERS_COLLECTION)

        uid = decoded_token["uid"]
        email = decoded_token.get("email", "")
        name = decoded_token.get("name", "")
        picture = decoded_token.get("picture", "")

        user_doc = users_ref.document(uid).get()

        user_data = {
            "uid": uid,
            "email": email,
            "name": name,
            "picture": picture,
            "provider": "google",
            "is_active": True,
            "last_login": datetime.utcnow(),
        }

        if user_doc.exists:
            users_ref.document(uid).update(
                {"last_login": datetime.utcnow(), "name": name, "picture": picture}
            )
            existing_data = user_doc.to_dict()
            existing_data.update(user_data)
            return existing_data

        user_data["created_at"] = datetime.utcnow()
        users_ref.document(uid).set(user_data)
        return user_data

    except Exception as e:
        print(f"Error al obtener/crear usuario: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor",
        )


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Crear token de acceso JWT."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verificar token JWT."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="No se pudieron validar las credenciales",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        uid: str = payload.get("sub")
        if uid is None:
            raise credentials_exception
        return GoogleTokenData(uid=uid)
    except JWTError:
        raise credentials_exception


def get_current_user(token_data: GoogleTokenData = Depends(verify_token)):
    """Obtener usuario actual basado en el token de Google."""
    try:
        db = get_firestore_client()
        users_ref = db.collection(USERS_COLLECTION)
        user_doc = users_ref.document(token_data.uid).get()

        if not user_doc.exists:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Usuario no encontrado")

        user_data = user_doc.to_dict()
        if not user_data.get("is_active", False):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Usuario inactivo")

        return user_data

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error al obtener usuario actual: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error interno del servidor",
        )


def authenticate_with_google(google_auth_request: GoogleAuthRequest):
    """Autenticar usuario con Google y devolver token de acceso."""
    try:
        decoded_token = verify_google_token(google_auth_request.id_token)
        user_data = get_or_create_user_from_google(decoded_token)

        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user_data["uid"], "email": user_data["email"]},
            expires_delta=access_token_expires,
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": {
                "uid": user_data["uid"],
                "email": user_data["email"],
                "name": user_data["name"],
                "picture": user_data["picture"],
                "provider": user_data["provider"],
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error en autenticación con Google: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error en la autenticación con Google",
        )


def logout_user(current_user: dict = Depends(get_current_user)):
    """Cerrar sesión del usuario (invalidar token en el cliente)."""
    try:
        return {"message": "Sesión cerrada exitosamente"}
    except Exception as e:
        print(f"Error al cerrar sesión: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error al cerrar sesión",
        )
