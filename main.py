import os

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.firebase import initialize_firestore
from app.routes import auth, fault_campaign, fault_injection, models, sessions, vhdl
from app.services.auth_service import initialize_default_users

app = FastAPI()

try:
    initialize_firestore()
    initialize_default_users()
    print("✅ Firestore inicializado correctamente")
except Exception as e:
    print(f"⚠️ Advertencia: No se pudo inicializar Firestore: {str(e)}")
    print("   Asegúrate de tener el archivo de credenciales de Firebase")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "¡FastAPI está corriendo!"}


app.include_router(auth.router)
app.include_router(models.router)
app.include_router(sessions.router)
app.include_router(vhdl.router)
app.include_router(fault_injection.router)
app.include_router(fault_campaign.router)


if __name__ == "__main__":
    host = "0.0.0.0"
    port = int(os.getenv("PORT", 8002))

    print(f"Iniciando servidor en http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
