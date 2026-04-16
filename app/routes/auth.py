from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import ACCESS_TOKEN_EXPIRE_MINUTES
from app.schemas.auth import GoogleAuthRequest, Token, UserCreate, UserLogin
from app.services.auth_service import (
    authenticate_user,
    create_access_token,
    create_user,
    get_current_user,
)
from app.services.google_auth_service import authenticate_with_google, logout_user
from app.services.google_auth_service import get_current_user as get_current_google_user

router = APIRouter(tags=["auth"])


@router.post("/auth/register", response_model=dict)
def register(user: UserCreate, current_user: dict = Depends(get_current_user)):
    """Registrar nuevo usuario."""
    try:
        new_user = create_user(user)
        return {
            "success": True,
            "message": "Usuario creado exitosamente",
            "user": new_user,
            "created_by": current_user.get("username", current_user.get("email", "unknown")),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auth/login", response_model=Token)
def login(user_credentials: UserLogin):
    """Iniciar sesión con credenciales locales."""
    user = authenticate_user(user_credentials.username, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales incorrectas",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "username": user["username"],
            "email": user["email"],
            "is_active": user["is_active"],
        },
    }


@router.get("/auth/me")
def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Obtener información del usuario actual."""
    return {
        "username": current_user.get("username"),
        "email": current_user.get("email"),
        "is_active": current_user.get("is_active", True),
    }


@router.post("/auth/google", response_model=dict)
def google_login(google_auth_request: GoogleAuthRequest):
    """Autenticación con Google."""
    try:
        result = authenticate_with_google(google_auth_request)
        return {
            "success": True,
            "message": "Autenticación con Google exitosa",
            "data": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en autenticación con Google: {str(e)}",
        )


@router.post("/auth/google/logout")
def google_logout(current_user: dict = Depends(get_current_google_user)):
    """Cerrar sesión de Google."""
    try:
        result = logout_user(current_user)
        return {
            "success": True,
            "message": "Sesión cerrada exitosamente",
            "data": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al cerrar sesión: {str(e)}",
        )


@router.get("/auth/google/me")
def get_current_google_user_info(current_user: dict = Depends(get_current_google_user)):
    """Obtener información del usuario actual autenticado con Google."""
    return {
        "uid": current_user.get("uid"),
        "email": current_user.get("email"),
        "name": current_user.get("name"),
        "picture": current_user.get("picture"),
        "provider": current_user.get("provider"),
        "is_active": current_user.get("is_active", True),
    }


@router.post("/auth/verify-token")
def verify_token_endpoint(current_user: dict = Depends(get_current_user)):
    """Verificar si el token es válido (soporta usuarios locales y Google)."""
    user_data = {
        "email": current_user.get("email"),
        "is_active": current_user.get("is_active", True),
    }

    if "username" in current_user:
        user_data["username"] = current_user["username"]

    if "name" in current_user:
        user_data["displayName"] = current_user["name"]

    if "picture" in current_user:
        user_data["photoURL"] = current_user["picture"]

    if "provider" in current_user:
        user_data["provider"] = current_user["provider"]

    return {"valid": True, "user": user_data}
