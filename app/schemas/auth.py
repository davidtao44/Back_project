from typing import Optional

from pydantic import BaseModel


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


class GoogleAuthRequest(BaseModel):
    id_token: str


class GoogleTokenData(BaseModel):
    uid: Optional[str] = None
    email: Optional[str] = None
