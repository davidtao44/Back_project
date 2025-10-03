from fastapi import FastAPI, HTTPException, Body, Depends, status, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel  # Añadir esta línea
import uvicorn
from typing import List
import tensorflow as tf
import os
import time
import base64
import io
from PIL import Image
import numpy as np
from datetime import timedelta
import shutil
import uuid
import json
import subprocess

# Importar desde nuestros módulos
from models import ImageToVHDLRequest, FaultInjectorRequest, FaultInjectorInferenceRequest, FaultInjectionConfig

# Importar la función de cuantización
from model_quantization import modify_and_save_weights

# Importar bibliotecas adicionales si no están ya importadas
import keras

# Importar inferencia manual
from fault_injection.manual_inference import ManualInference

# Importar módulos de hardware VHDL
from vhdl_hardware.hardware_fault_injector import HardwareFaultInjector
from vhdl_hardware.vhdl_weight_modifier import VHDLWeightModifier
from vhdl_hardware.vivado_controller import VivadoController

def sanitize_for_json(obj):
    """
    Recursivamente limpia un objeto para asegurar que sea serializable a JSON.
    Reemplaza inf, -inf, y NaN con valores válidos.
    """
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_json(item) for item in obj)
    elif isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    elif isinstance(obj, float):
        if np.isnan(obj):
            return 0.0
        elif np.isinf(obj):
            return 1.0 if obj > 0 else 0.0
        else:
            return obj
    elif isinstance(obj, np.ndarray):
        # Convertir arrays numpy a listas y limpiar
        return sanitize_for_json(obj.tolist())
    elif hasattr(obj, '__dict__'):
        # Para objetos personalizados, convertir a dict
        return sanitize_for_json(obj.__dict__)
    else:
        # Para otros tipos, intentar convertir a string
        try:
            return str(obj)
        except:
            return "unknown_type"

# Importar módulos de autenticación
from auth import (
    UserCreate, UserLogin, Token, authenticate_user, create_access_token,
    get_current_user, create_user, ACCESS_TOKEN_EXPIRE_MINUTES,
    initialize_default_users
)

# Importar autenticación con Google
from google_auth import (
    GoogleAuthRequest, authenticate_with_google, logout_user,
    get_current_user as get_current_google_user
)

# Importar configuración de Firebase
from firebase_config import initialize_firestore

app = FastAPI()

# Inicializar Firestore y usuarios por defecto
try:
    initialize_firestore()
    initialize_default_users()
    print("✅ Firestore inicializado correctamente")
except Exception as e:
    print(f"⚠️ Advertencia: No se pudo inicializar Firestore: {str(e)}")
    print("   Asegúrate de tener el archivo de credenciales de Firebase")

# Configurar CORS para permitir peticiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # permite toda ruta
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados
)



@app.get("/")
def read_root():
    return {"message": "¡FastAPI está corriendo!"}

# Endpoints de autenticación
@app.post("/auth/register", response_model=dict)
def register(user: UserCreate, current_user: dict = Depends(get_current_user)):
    """Registrar nuevo usuario"""
    try:
        new_user = create_user(user)
        return {
            "success": True,
            "message": "Usuario creado exitosamente",
            "user": new_user,
            "created_by": current_user.get("username", current_user.get("email", "unknown"))
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cleanup_sessions/")
def cleanup_old_sessions(max_age_hours: int = 24, current_user: dict = Depends(get_current_user)):
    """Limpia las carpetas de sesión más antiguas que max_age_hours"""
    try:
        # Directorios de sesión a limpiar
        session_dirs = [
            os.path.join(os.path.dirname(__file__), "layer_outputs"),
            os.path.join(os.path.dirname(__file__), "vhdl_outputs"),
            os.path.join(os.path.dirname(__file__), "model_weights_outputs")
        ]
        
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cleaned_sessions = []
        total_cleaned = 0
        
        for base_dir in session_dirs:
            if os.path.exists(base_dir):
                for session_folder in os.listdir(base_dir):
                    session_path = os.path.join(base_dir, session_folder)
                    if os.path.isdir(session_path):
                        # Verificar la edad de la carpeta
                        folder_age = current_time - os.path.getctime(session_path)
                        if folder_age > max_age_seconds:
                            shutil.rmtree(session_path)
                            cleaned_sessions.append(f"{os.path.basename(base_dir)}/{session_folder}")
                            total_cleaned += 1
        
        if total_cleaned == 0:
            return {"message": "No hay carpetas de sesión antiguas para limpiar"}
        
        return {
            "message": f"Se limpiaron {total_cleaned} sesiones antiguas",
            "cleaned_sessions": cleaned_sessions
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al limpiar sesiones: {str(e)}")

@app.get("/list_sessions/")
def list_active_sessions(current_user: dict = Depends(get_current_user)):
    """Lista todas las sesiones activas con información básica"""
    try:
        # Directorios de sesión a listar
        session_dirs = [
            ("layer_outputs", os.path.join(os.path.dirname(__file__), "layer_outputs")),
            ("vhdl_outputs", os.path.join(os.path.dirname(__file__), "vhdl_outputs")),
            ("model_weights_outputs", os.path.join(os.path.dirname(__file__), "model_weights_outputs"))
        ]
        
        sessions = []
        current_time = time.time()
        
        for dir_type, base_dir in session_dirs:
            if os.path.exists(base_dir):
                for session_folder in os.listdir(base_dir):
                    session_path = os.path.join(base_dir, session_folder)
                    if os.path.isdir(session_path):
                        # Obtener información de la sesión
                        creation_time = os.path.getctime(session_path)
                        age_hours = (current_time - creation_time) / 3600
                        
                        # Contar archivos en la sesión
                        file_count = 0
                        for root, dirs, files in os.walk(session_path):
                            file_count += len(files)
                        
                        sessions.append({
                            "session_id": session_folder,
                            "session_type": dir_type,
                            "created_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(creation_time)),
                            "age_hours": round(age_hours, 2),
                            "file_count": file_count
                        })
        
        # Ordenar por fecha de creación (más recientes primero)
        sessions.sort(key=lambda x: x['created_at'], reverse=True)
        
        return {
            "total_sessions": len(sessions),
            "sessions": sessions
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al listar sesiones: {str(e)}")

@app.post("/auth/login", response_model=Token)
def login(user_credentials: UserLogin):
    """Iniciar sesión"""
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
            "is_active": user["is_active"]
        }
    }

@app.get("/auth/me")
def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """Obtener información del usuario actual"""
    return {
        "username": current_user.get("username"),
        "email": current_user.get("email"),
        "is_active": current_user.get("is_active", True)
    }

@app.post("/auth/google", response_model=dict)
def google_login(google_auth_request: GoogleAuthRequest):
    """Autenticación con Google"""
    try:
        result = authenticate_with_google(google_auth_request)
        return {
            "success": True,
            "message": "Autenticación con Google exitosa",
            "data": result
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en autenticación con Google: {str(e)}"
        )

@app.post("/auth/google/logout")
def google_logout(current_user: dict = Depends(get_current_google_user)):
    """Cerrar sesión de Google"""
    try:
        result = logout_user(current_user)
        return {
            "success": True,
            "message": "Sesión cerrada exitosamente",
            "data": result
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error al cerrar sesión: {str(e)}"
        )

@app.get("/auth/google/me")
def get_current_google_user_info(current_user: dict = Depends(get_current_google_user)):
    """Obtener información del usuario actual autenticado con Google"""
    return {
        "uid": current_user.get("uid"),
        "email": current_user.get("email"),
        "name": current_user.get("name"),
        "picture": current_user.get("picture"),
        "provider": current_user.get("provider"),
        "is_active": current_user.get("is_active", True)
    }

@app.post("/auth/verify-token")
def verify_token_endpoint(current_user: dict = Depends(get_current_user)):
    """Verificar si el token es válido"""
    # Manejar tanto usuarios normales como usuarios de Google
    user_data = {
        "email": current_user.get("email"),
        "is_active": current_user.get("is_active", True)
    }
    
    # Para usuarios normales (tienen username)
    if "username" in current_user:
        user_data["username"] = current_user["username"]
    
    # Para usuarios de Google (tienen name, picture, etc.)
    if "name" in current_user:
        user_data["displayName"] = current_user["name"]
    
    if "picture" in current_user:
        user_data["photoURL"] = current_user["picture"]
    
    if "provider" in current_user:
        user_data["provider"] = current_user["provider"]
    
    return {
        "valid": True,
        "user": user_data
    }

@app.get("/models/")
def get_models(current_user: dict = Depends(get_current_user)):
    """Endpoint para obtener la lista de modelos disponibles - Requiere autenticación"""
    try:
        models_dir = "models"
        if not os.path.exists(models_dir):
            return {"models": []}
        
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
        models_info = []
        
        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            # Obtener tiempo de creación y tamaño
            creation_time = os.path.getctime(model_path)
            size_kb = os.path.getsize(model_path) / 1024
            
            # Cargar modelo para obtener información de resumen
            try:
                model = tf.keras.models.load_model(model_path)
                # Obtener información de capas
                layers_info = []
                for layer in model.layers:
                    layer_info = {
                        "name": layer.name,
                        "type": layer.__class__.__name__,
                    }
                    if hasattr(layer, 'units'):
                        layer_info["units"] = layer.units
                    if hasattr(layer, 'filters'):
                        layer_info["filters"] = layer.filters
                    layers_info.append(layer_info)
                
                models_info.append({
                    "filename": model_file,
                    "path": model_path,
                    "created_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(creation_time)),
                    "size_kb": round(size_kb, 2),
                    "layers": layers_info
                })
            except Exception as e:
                # Si hay un error al cargar el modelo, incluir información básica
                models_info.append({
                    "filename": model_file,
                    "path": model_path,
                    "created_at": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(creation_time)),
                    "size_kb": round(size_kb, 2),
                    "error": str(e)
                })
        
        return {"models": models_info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list_models/")
def list_models(current_user: dict = Depends(get_current_user)):
    """Endpoint legacy para compatibilidad - Requiere autenticación"""
    return get_models(current_user)

@app.post("/upload_model/")
async def upload_model(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    """
    Endpoint para subir modelos CNN preentrenados
    """
    try:
        # Verificar que el archivo sea un modelo válido
        if not file.filename.endswith(('.h5', '.keras')):
            raise HTTPException(
                status_code=400, 
                detail="Solo se permiten archivos .h5 o .keras"
            )
        
        # Crear directorio models si no existe
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        
        # Generar nombre único para evitar conflictos
        timestamp = int(time.time())
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(models_dir, filename)
        
        # Guardar el archivo
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Verificar que el archivo se puede cargar como modelo
        try:
            model = tf.keras.models.load_model(file_path)
            model_info = {
                "name": filename,
                "path": file_path,
                "layers": len(model.layers),
                "parameters": model.count_params(),
                "input_shape": str(model.input_shape),
                "output_shape": str(model.output_shape)
            }
        except Exception as e:
            # Si no se puede cargar, eliminar el archivo
            os.remove(file_path)
            raise HTTPException(
                status_code=400,
                detail=f"El archivo no es un modelo válido: {str(e)}"
            )
        
        return {
            "message": "Modelo subido exitosamente",
            "model_info": model_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al subir el modelo: {str(e)}"
        )

@app.post("/delete_models/")
def delete_models(model_paths: List[str] = Body(...), current_user: dict = Depends(get_current_user)):
    try:
        deleted_models = []
        errors = []
        
        for model_path in model_paths:
            try:
                if os.path.exists(model_path):
                    os.remove(model_path)
                    deleted_models.append(os.path.basename(model_path))
                else:
                    errors.append(f"Modelo no encontrado: {os.path.basename(model_path)}")
            except Exception as e:
                errors.append(f"Error al eliminar {os.path.basename(model_path)}: {str(e)}")
        
        return {
            "success": len(deleted_models) > 0,
            "deleted_models": deleted_models,
            "errors": errors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/quantize_model/")
def quantize_model(model_path: str = Body(...), multiplication_factor: int = Body(100), current_user: dict = Depends(get_current_user)):
    try:
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Modelo no encontrado: {os.path.basename(model_path)}")
        
        # Generar nombre para el modelo cuantizado
        base_name = os.path.basename(model_path)
        name_parts = os.path.splitext(base_name)
        quantized_filename = f"{name_parts[0]}_quantized_{multiplication_factor}{name_parts[1]}"
        save_path = os.path.join("models", quantized_filename)
        
        # Cuantizar el modelo
        result_path = modify_and_save_weights(model_path, save_path, multiplication_factor)
        
        return {
            "success": True,
            "message": f"Modelo cuantizado con factor {multiplication_factor} exitosamente",
            "original_model": os.path.basename(model_path),
            "quantized_model": os.path.basename(result_path),
            "quantized_model_path": result_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/convert_image_to_vhdl/")
async def convert_image_to_vhdl(request: ImageToVHDLRequest, current_user: dict = Depends(get_current_user)):
    try:
        # Decodificar la imagen desde base64
        image_data = request.image_data
        if "," in image_data:
            # Eliminar el prefijo de datos URI si existe
            image_data = image_data.split(",")[1]
        
        # Decodificar base64 a bytes
        image_bytes = base64.b64decode(image_data)
        
        # Abrir la imagen con PIL
        img = Image.open(io.BytesIO(image_bytes))
        
        # Redimensionar la imagen
        img = img.resize((request.width, request.height))
        
        # Convertir a escala de grises
        img = img.convert('L')
        
        # Convertir a matriz numpy
        pixel_matrix = np.array(img)
        
        # Preparar matrices de salida
        decimal_matrix = pixel_matrix.tolist()
        hex_matrix = []
        vhdl_matrix = []
        
        # Convertir a hexadecimal y formato VHDL
        for row in decimal_matrix:
            hex_row = []
            vhdl_row = []
            for pixel in row:
                hex_value = format(pixel, '02x')
                hex_row.append(hex_value)
                vhdl_row.append(f'x"{hex_value}"')
            hex_matrix.append(hex_row)
            vhdl_matrix.append(vhdl_row)
        
        # Generar código VHDL
        vhdl_code = generate_vhdl_code(vhdl_matrix, request.width, request.height)
        
        # Crear directorio de sesión único para el usuario
        session_id = f"user_{current_user.get('uid', 'anonymous')}_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        session_dir = os.path.join(os.path.dirname(__file__), "vhdl_outputs", session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Guardar el código VHDL en un archivo de texto con la estructura de Memoria_Imagen.vhd
        output_filename = f"Memoria_Imagen_{session_id}.vhdl.txt"
        output_path = os.path.join(session_dir, output_filename)
        with open(output_path, "w") as f:
            f.write(vhdl_code)
        
        return {
            "success": True,
            "decimal_matrix": decimal_matrix,
            "hex_matrix": hex_matrix,
            "vhdl_code": vhdl_code,
            "file_path": output_filename,  # Retornar solo el nombre del archivo
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def generate_vhdl_code(vhdl_matrix, width, height):
    # Generar el código VHDL con la estructura de Memoria_Imagen.vhd
    vhdl_code = [
        "library ieee; ",
        "use ieee.std_logic_1164.all; ",
        "use ieee.numeric_std.all; ",
        "",
        "entity Memoria_Imagen is",
        "",
        "	generic (",
        "		data_width  : natural := 8; ",
        "	   addr_length : natural := 10	-- 1024 pos mem",
        "		); ",
        "	",
        "	port ( ",
        "		clk      :  in std_logic;",
        "--		rst : in std_logic;",
        "		address  :  in std_logic_vector(addr_length-1 downto 0); ",
        "		data_out :  out std_logic_vector(data_width-1  downto 0) ",
        "		);",
        "		",
        "end Memoria_Imagen;",
        "",
        "",
        "architecture synth of Memoria_Imagen is",
        "",
        "	constant mem_size : natural := 2**addr_length; 	",
        "	type mem_type is array (0 to mem_size-1) of std_logic_vector (data_width-1 downto 0); ",
        "",
        "constant mem : mem_type := ("
    ]
    
    # Formatear la matriz de píxeles en el formato de Memoria_Imagen.vhd
    # Aplanar la matriz 2D en una matriz 1D para el formato de Memoria_Imagen.vhd
    flat_matrix = []
    for row in vhdl_matrix:
        flat_matrix.extend(row)
    
    # Agregar los valores en grupos de 32 por línea
    for i in range(0, len(flat_matrix), 32):
        chunk = flat_matrix[i:i+32]
        line = "\t\t" + ", ".join(chunk)
        if i + 32 < len(flat_matrix):
            line += ","
        vhdl_code.append(line)
    
    # Cerrar la definición de la matriz y agregar el resto del código
    vhdl_code.extend([
        "	);",
        "",
        "	",
        "begin ",
        "",
        "	rom : process (clk) ",
        "	begin",
        "	   ",
        "----	   if (rising_edge(Clk)) then",
        "--		  if rst = '1' then",
        "--				-- Reset the counter to 0",
        "--				data_out <= ((others=> '0'));",
        "--		  end if;",
        "	   ",
        "		if rising_edge(clk) then ",
        "			data_out <= mem(to_integer(unsigned(address))); ",
        "		end if; ",
        "	end process rom; ",
        "",
        "end architecture synth;"
    ])
    
    return "\n".join(vhdl_code)

def to_binary_c2(value, bits=8):
    """Convierte un valor a binario en complemento a 2."""
    if value < 0:
        value = (1 << bits) + value
    return format(value, f'0{bits}b')

def extract_filters(model, output_dir, bits_value=8):
    """Extrae los filtros y sesgos de cada capa convolucional y los guarda en archivos .txt."""
    # Crear el directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filter_files = []
    
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            filters, biases = layer.get_weights()
            num_filters = filters.shape[-1]  # Número de filtros
            num_channels = filters.shape[-2]  # Número de canales de entrada

            # Crear un archivo de texto para la capa actual
            layer_filename = os.path.join(output_dir, f"layer_{i+1}_filters.txt")
            filter_files.append(os.path.basename(layer_filename))
            
            with open(layer_filename, "w") as file:
                # Escribir los filtros en el archivo
                file.write(f"Filtros de la capa convolucional {i+1}:\n")
                for j in range(num_filters):  # Recorrer cada filtro
                    for c in range(num_channels):  # Recorrer cada canal
                        filter_matrix = filters[:, :, c, j]  # Obtener el filtro para el canal c
                        file.write(f"constant FMAP_{c+1}_{j+1}: FILTER_TYPE:= (\n")
                        for k in range(filter_matrix.shape[0]):  # Filas del filtro
                            row = ["\"" + to_binary_c2(int(filter_matrix[k, l]), bits=bits_value) + "\"" 
                                   for l in range(filter_matrix.shape[1])]
                            file.write("    (" + ",".join(row) + ")" + ("," if k < filter_matrix.shape[0] - 1 else "") + "\n")
                        file.write(");\n\n")

                # Escribir los sesgos en el archivo
                file.write(f"Sesgos de la capa convolucional {i+1}:\n")
                for idx, bias in enumerate(biases):
                    bias_bin = to_binary_c2(int(bias), bits=bits_value)  # Mantener 24 bits para sesgos o usar bits_value
                    file.write(f"constant BIAS_VAL_{idx+1}: signed (BIASES_SIZE-1 downto 0) := \"{bias_bin}\";\n")
                file.write("\n")
    
    return filter_files

def extract_dense_layers(model, output_dir, bits_value=8):
    """Extrae los pesos y sesgos de las capas densas y los guarda en archivos .txt."""
    # Crear el directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dense_files = []
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            pesos = layer.get_weights()[0]  # Pesos
            biases = layer.get_weights()[1]  # Biases

            num_filas, num_columnas = pesos.shape  # (Entradas, Neuronas)

            # Guardar pesos en formato VHDL (recorriendo por columnas)
            pesos_filename = os.path.join(output_dir, f"{layer.name}_pesos.txt")
            dense_files.append(os.path.basename(pesos_filename))
            
            with open(pesos_filename, "w") as f_pesos:
                for j in range(num_columnas):  # Recorrer por columnas primero
                    for i in range(num_filas):  # Luego recorrer filas
                        bin_value = to_binary_c2(int(pesos[i, j]), bits=bits_value)  # Usar bits_value
                        f_pesos.write(f'constant FMAP_{j+1}_{i+1}: signed(WEIGHT_SIZE-1 downto 0) := "{bin_value}";\n')

            # Guardar biases en formato VHDL
            biases_filename = os.path.join(output_dir, f"{layer.name}_biases.txt")
            dense_files.append(os.path.basename(biases_filename))
            
            with open(biases_filename, "w") as f_biases:
                for i, bias in enumerate(biases, start=1):  # Iniciar en 1
                    bin_value = to_binary_c2(int(bias), bits=bits_value)  # Usar bits_value
                    f_biases.write(f'constant BIAS_VAL_{i}: signed(BIASES_SIZE-1 downto 0) := "{bin_value}";\n')
    
    return dense_files

# Crear un nuevo modelo para la solicitud de extracción de pesos
class ModelWeightsRequest(BaseModel):
    model_path: str
    output_dir: str = "model_weights"
    bits_value: int = 8  # Valor predeterminado de 8 bits

# Endpoint para extraer pesos y sesgos del modelo
@app.post("/extract_model_weights/")
def extract_model_weights(request: ModelWeightsRequest, current_user: dict = Depends(get_current_user)):
    try:
        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=404, detail=f"Modelo no encontrado: {os.path.basename(request.model_path)}")
        
        # Crear directorio de sesión único para el usuario
        session_id = f"user_{current_user.get('uid', 'anonymous')}_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        session_dir = os.path.join(os.path.dirname(__file__), "model_weights_outputs", session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Usar el directorio de sesión como output_dir
        output_dir = session_dir
        
        # Cargar el modelo
        model = keras.models.load_model(request.model_path)
        
        # Establecer el valor de bits para la conversión
        bits_value = request.bits_value
        
        # Extraer filtros y sesgos de capas convolucionales
        conv_files = extract_filters(model, output_dir, bits_value)
        
        # Extraer pesos y sesgos de capas densas
        dense_files = extract_dense_layers(model, output_dir, bits_value)
        
        # Combinar todos los archivos generados
        generated_files = conv_files + dense_files
        
        # Convertir rutas absolutas a nombres de archivo relativos
        relative_files = [os.path.basename(file_path) for file_path in generated_files]
        
        return {
            "success": True,
            "message": f"Pesos y sesgos extraídos exitosamente con {bits_value} bits",
            "model": os.path.basename(request.model_path),
            "output_dir": os.path.basename(output_dir),
            "files": relative_files,
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint para descargar archivos generados
@app.get("/download_file/")
def download_file(file_path: str, t: str = None):  # Agregar parámetro t para cache busting (opcional)
    try:
        # Si es una ruta relativa, buscar en todas las carpetas de sesión
        if not os.path.isabs(file_path):
            # Directorios donde buscar archivos
            search_dirs = [
                os.path.join(os.path.dirname(__file__), "layer_outputs"),
                os.path.join(os.path.dirname(__file__), "vhdl_outputs"),
                os.path.join(os.path.dirname(__file__), "model_weights_outputs")
            ]
            
            found_file = None
            
            # Si se proporciona un session_id (parámetro t), buscar primero en esa sesión específica
            if t:
                for base_dir in search_dirs:
                    if os.path.exists(base_dir):
                        session_path = os.path.join(base_dir, t)
                        if os.path.isdir(session_path):
                            potential_file = os.path.join(session_path, file_path)
                            if os.path.exists(potential_file):
                                found_file = potential_file
                                break
                if found_file:
                    pass  # Ya encontramos el archivo en la sesión específica
            
            # Si no se encontró en la sesión específica, buscar en todas las sesiones (ordenadas por fecha de modificación)
            if found_file is None:
                session_files = []
                for base_dir in search_dirs:
                    if os.path.exists(base_dir):
                        for session_folder in os.listdir(base_dir):
                            session_path = os.path.join(base_dir, session_folder)
                            if os.path.isdir(session_path):
                                potential_file = os.path.join(session_path, file_path)
                                if os.path.exists(potential_file):
                                    # Agregar el archivo con su tiempo de modificación
                                    mtime = os.path.getmtime(potential_file)
                                    session_files.append((potential_file, mtime))
                
                # Ordenar por tiempo de modificación (más reciente primero) y tomar el primero
                if session_files:
                    session_files.sort(key=lambda x: x[1], reverse=True)
                    found_file = session_files[0][0]
            
            # Fallback: buscar directamente en los directorios base
            if found_file is None:
                for base_dir in search_dirs:
                    fallback_path = os.path.join(base_dir, file_path)
                    if os.path.exists(fallback_path):
                        found_file = fallback_path
                        break
            
            # Fallback adicional: buscar en el directorio raíz del proyecto
            if found_file is None:
                root_fallback = os.path.join(os.path.dirname(__file__), file_path)
                if os.path.exists(root_fallback):
                    found_file = root_fallback
            
            full_path = found_file
        else:
            full_path = file_path
        
        # Verificar que el archivo existe
        if not full_path or not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {os.path.basename(file_path)}")
        
        # Determinar el tipo de archivo y el media type
        filename = os.path.basename(full_path)
        file_extension = os.path.splitext(filename)[1].lower()
        
        if file_extension == '.png':
            media_type = 'image/png'
        elif file_extension == '.xlsx':
            media_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        elif file_extension == '.jpg' or file_extension == '.jpeg':
            media_type = 'image/jpeg'
        else:
            media_type = 'application/octet-stream'
        
        # Crear FileResponse con headers para prevenir caché
        response = FileResponse(
            path=full_path,
            media_type=media_type,
            filename=filename
        )
        
        # Agregar headers para prevenir caché del navegador
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fault_injector/configure/")
async def configure_fault_injection(
    fault_config: FaultInjectionConfig,
    current_user: dict = Depends(get_current_user)
):
    """
    Endpoint para configurar la inyección de fallos
    """
    try:
        # Validar la configuración
        if not fault_config.layers:
            raise HTTPException(status_code=400, detail="Debe especificar al menos una capa para inyección de fallos")
        
        # Validar que los tipos de fallos sean válidos
        valid_fault_types = ["bit_flip", "stuck_at_0", "stuck_at_1", "random_noise"]
        for layer_name, layer_config in fault_config.layers.items():
            if layer_config.fault_type not in valid_fault_types:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Tipo de fallo inválido '{layer_config.fault_type}' para la capa '{layer_name}'. Tipos válidos: {valid_fault_types}"
                )
            
            # Validar parámetros específicos
            if layer_config.fault_rate < 0 or layer_config.fault_rate > 1:
                raise HTTPException(
                    status_code=400,
                    detail=f"La tasa de fallos debe estar entre 0 y 1 para la capa '{layer_name}'"
                )
        
        return {
            "success": True,
            "message": "Configuración de inyección de fallos validada correctamente",
            "config": fault_config.dict(),
            "layers_configured": len(fault_config.layers),
            "enabled": fault_config.enabled
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al configurar inyección de fallos: {str(e)}")

@app.post("/fault_injector/inference/")
async def fault_injector_inference(
    file: UploadFile = File(...), 
    model_path: str = Body(...), 
    fault_config: str = Body(None),
    current_user: dict = Depends(get_current_user)
):
    """
    Endpoint para realizar inferencia manual con FaultInjector y configuración de fallos
    """
    try:
        # Validar que el archivo sea una imagen
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
        
        # Leer los datos de la imagen
        image_data = await file.read()
        
        # Validar que el modelo existe
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Modelo no encontrado")
        
        # Procesar configuración de fallos si se proporciona
        fault_config_dict = None
        if fault_config:
            try:
                import json
                fault_config_dict = json.loads(fault_config)
                print(f"🔧 DEBUG: Configuración de fallos recibida: {fault_config_dict}")
                
                # Verificar si es la nueva estructura combinada
                if 'activation_faults' in fault_config_dict or 'weight_faults' in fault_config_dict:
                    print("🔧 DEBUG: Detectada configuración combinada de fallos")
                else:
                    print("🔧 DEBUG: Configuración de fallos legacy (solo activaciones)")
                    
            except json.JSONDecodeError:
                print(f"❌ ERROR: Configuración de fallos inválida: {fault_config}")
                raise HTTPException(status_code=400, detail="Configuración de fallos inválida")
        else:
            print("ℹ️ DEBUG: No se recibió configuración de fallos")
        
        # Crear instancia de ManualInference con ruta absoluta y session_id único
        output_dir = os.path.join(os.path.dirname(__file__), "layer_outputs")
        session_id = f"user_{current_user.get('uid', 'anonymous')}_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        manual_inference = ManualInference(
            model_path, 
            output_dir=output_dir, 
            session_id=session_id,
            fault_config=fault_config_dict
        )
        
        # Realizar inferencia manual
        results = manual_inference.perform_manual_inference(image_data)
        
        # Convertir rutas absolutas a nombres de archivo para el frontend
        excel_filenames = [os.path.basename(f) for f in results["excel_files"] if f]
        image_filenames = [os.path.basename(f) for f in results["image_files"] if f]
        
        # Verificar si hay errores de overflow/underflow
        final_prediction = results["final_prediction"]
        if not final_prediction.get("success", True):
            # Hay errores numéricos - devolver información detallada del error
            response_data = {
                "success": False,
                "error_type": "numerical_overflow_underflow",
                "error_details": final_prediction["error"],
                "session_id": results["session_id"],
                "layer_outputs": results["layer_outputs"],
                "excel_files": excel_filenames,
                "image_files": image_filenames,
                "model_used": os.path.basename(model_path),
                "fault_injection": results["fault_injection"],
                "message": "Error numérico detectado durante la inferencia"
            }
        else:
            # Inferencia exitosa
            response_data = {
                "success": True,
                "session_id": results["session_id"],
                "predicted_class": final_prediction["predicted_class"],
                "confidence": final_prediction["confidence"],
                "all_probabilities": final_prediction["all_probabilities"],
                "layer_outputs": results["layer_outputs"],
                "excel_files": excel_filenames,
                "image_files": image_filenames,
                "model_used": os.path.basename(model_path),
                "fault_injection": results["fault_injection"],
                "message": "Inferencia manual completada exitosamente"
            }
        
        # Sanitizar la respuesta para asegurar compatibilidad con JSON
        return sanitize_for_json(response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la inferencia: {str(e)}")

# ==================== ENDPOINTS DE VHDL HARDWARE ====================

@app.get("/vhdl/supported_faults/")
def get_supported_faults():
    """
    Obtiene información sobre los tipos de fallos soportados para inyección en hardware VHDL
    """
    try:
        # Estructura que espera el frontend
        supported_faults = {
            "filter_targets": [
                {"name": "filter_0", "description": "Filtro 0 - Primera capa convolucional"},
                {"name": "filter_1", "description": "Filtro 1 - Primera capa convolucional"},
                {"name": "filter_2", "description": "Filtro 2 - Primera capa convolucional"},
                {"name": "filter_3", "description": "Filtro 3 - Primera capa convolucional"},
                {"name": "filter_4", "description": "Filtro 4 - Primera capa convolucional"},
                {"name": "filter_5", "description": "Filtro 5 - Primera capa convolucional"}
            ],
            "bias_targets": [
                {"name": "bias_0", "description": "Sesgo 0 - Primera capa convolucional"},
                {"name": "bias_1", "description": "Sesgo 1 - Primera capa convolucional"},
                {"name": "bias_2", "description": "Sesgo 2 - Primera capa convolucional"},
                {"name": "bias_3", "description": "Sesgo 3 - Primera capa convolucional"},
                {"name": "bias_4", "description": "Sesgo 4 - Primera capa convolucional"},
                {"name": "bias_5", "description": "Sesgo 5 - Primera capa convolucional"}
            ],
            "fault_types": [
                {"name": "stuck_at_0", "description": "Forzar bit a valor 0"},
                {"name": "stuck_at_1", "description": "Forzar bit a valor 1"}
            ]
        }
        
        return supported_faults
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener fallos soportados: {str(e)}")

@app.get("/vhdl/validate_vivado/")
def validate_vivado_installation(vivado_path: str = None):
    """
    Valida si Vivado está instalado y accesible en la ruta especificada
    """
    try:
        vivado_controller = VivadoController(vivado_path or "vivado")
        is_valid = vivado_controller.verify_vivado_installation()
        
        return {
            "status": "success",
            "vivado_valid": is_valid,
            "vivado_path": vivado_path or "default",
            "message": "Validación de Vivado completada" if is_valid else "Vivado no encontrado o no válido"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "vivado_valid": False,
            "vivado_path": vivado_path or "default",
            "message": f"Error al validar Vivado: {str(e)}"
        }



@app.post("/vhdl/inject_faults/")
async def inject_vhdl_faults(
    filter_faults: str = Form(...),
    bias_faults: str = Form(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Endpoint simplificado para modificar pesos en archivo VHDL específico y ejecutar simulación
    """
    print(f"🔍 DEBUG: Recibidos parámetros - filter_faults: {filter_faults}, bias_faults: {bias_faults}")
    print(f"🔍 DEBUG: Usuario actual: {current_user}")
    
    try:
        # Archivo VHDL fijo
        vhdl_file_path = "/home/davidgonzalez/Documentos/David_2025/4_CONV1_SAB_STUCKAT_DEC_RAM_TB/CONV1_SAB_STUCKAT_DEC_RAM.srcs/sources_1/new/CONV1_SAB_STUCK_DECOS.vhd"
        print(f"🔍 DEBUG: Verificando archivo VHDL: {vhdl_file_path}")
        
        # Validar que el archivo existe
        if not os.path.exists(vhdl_file_path):
            print(f"❌ ERROR: Archivo VHDL no encontrado: {vhdl_file_path}")
            raise HTTPException(status_code=404, detail=f"Archivo VHDL no encontrado: {vhdl_file_path}")
        
        print("✅ Archivo VHDL encontrado")
        
        # Parsear configuraciones de fallos
        try:
            print(f"🔍 DEBUG: Parseando filter_faults: {filter_faults}")
            filter_config = json.loads(filter_faults) if filter_faults else {}
            print(f"🔍 DEBUG: Parseando bias_faults: {bias_faults}")
            bias_config = json.loads(bias_faults) if bias_faults else {}
            print(f"✅ Configuraciones parseadas - filter: {filter_config}, bias: {bias_config}")
        except json.JSONDecodeError as e:
            print(f"❌ ERROR: Error parseando JSON: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error parseando configuración JSON: {str(e)}")
        
        # Crear directorio temporal para respaldo
        temp_dir = f"/tmp/vhdl_backup_{uuid.uuid4()}"
        os.makedirs(temp_dir, exist_ok=True)
        print(f"✅ Directorio temporal creado: {temp_dir}")
        
        # Crear respaldo del archivo original
        backup_path = os.path.join(temp_dir, "CONV1_SAB_STUCK_DECOS_backup.vhd")
        shutil.copy2(vhdl_file_path, backup_path)
        print(f"✅ Respaldo creado: {backup_path}")

        # Leer contenido del archivo VHDL
        with open(vhdl_file_path, 'r', encoding='utf-8') as f:
            vhdl_content = f.read()
        print("✅ Contenido del archivo VHDL leído")

        # Preparar configuración de fallos
        fault_config = {
            'filter_faults': filter_config,
            'bias_faults': bias_config
        }

        # Modificar el contenido del archivo VHDL
        try:
            modified_content = modify_vhdl_weights_and_bias(vhdl_content, fault_config)
            print("✅ Contenido del archivo VHDL modificado")
            
            modification_results = {
                "status": "success",
                "message": "Pesos y bias modificados correctamente",
                "filter_modifications": len(filter_config),
                "bias_modifications": len(bias_config)
            }
        except Exception as mod_error:
            print(f"❌ ERROR modificando VHDL: {str(mod_error)}")
            modification_results = {
                "status": "error",
                "message": f"Error modificando VHDL: {str(mod_error)}"
            }
            modified_content = vhdl_content  # Usar contenido original si falla

        # Escribir el archivo modificado
        with open(vhdl_file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        print("✅ Archivo VHDL modificado guardado")

        # Ejecutar simulación
        simulation_script = "/home/davidgonzalez/Documentos/David_2025/4_CONV1_SAB_STUCKAT_DEC_RAM_TB/CONV1_SAB_STUCKAT_DEC_RAM.sim/sim_1/behav/xsim/simulate.sh"
        simulation_results = {}
        csv_processing_results = {}
        
        if os.path.exists(simulation_script):
            print(f"🔍 Ejecutando simulación: {simulation_script}")
            try:
                # Cambiar al directorio de simulación
                sim_dir = os.path.dirname(simulation_script)
                
                # Ejecutar simulación con timeout
                result = subprocess.run(
                    ["bash", simulation_script],
                    cwd=sim_dir,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minutos de timeout
                )
                
                if result.returncode == 0:
                    print("✅ Simulación ejecutada exitosamente")
                    simulation_results = {
                        "status": "success",
                        "message": "Simulación completada exitosamente",
                        "output": result.stdout[-1000:] if result.stdout else "",  # Últimos 1000 caracteres
                        "errors": result.stderr[-500:] if result.stderr else ""    # Últimos 500 caracteres de errores
                    }
                    
                    # Procesar archivos CSV después de la simulación exitosa
                    print("🔍 Buscando archivos CSV para procesar...")
                    csv_files = []
                    for file in os.listdir(sim_dir):
                        if file.endswith('.csv') and 'Conv1' in file:
                            csv_files.append(os.path.join(sim_dir, file))
                    
                    if csv_files:
                        print(f"✅ Encontrados {len(csv_files)} archivos CSV para procesar")
                        # Importar el procesador CSV
                        from vhdl_hardware.csv_processor import CSVProcessor
                        
                        csv_processor = CSVProcessor()
                        processed_results = []
                        
                        for csv_file in csv_files:
                            try:
                                print(f"🔄 Procesando archivo CSV: {csv_file}")
                                result = csv_processor.process_simulation_csv(csv_file)
                                processed_results.append({
                                    "file": csv_file,
                                    "result": result
                                })
                                print(f"✅ Archivo CSV procesado exitosamente: {csv_file}")
                            except Exception as csv_error:
                                print(f"❌ Error procesando CSV {csv_file}: {str(csv_error)}")
                                processed_results.append({
                                    "file": csv_file,
                                    "error": str(csv_error)
                                })
                        
                        csv_processing_results = {
                            "status": "success",
                            "processed_files": len(processed_results),
                            "results": processed_results
                        }
                    else:
                        print("⚠️ No se encontraron archivos CSV para procesar")
                        csv_processing_results = {
                            "status": "warning",
                            "message": "No se encontraron archivos CSV para procesar"
                        }
                        
                else:
                    print(f"❌ ERROR en simulación (código: {result.returncode})")
                    simulation_results = {
                        "status": "error",
                        "message": f"Simulación falló con código {result.returncode}",
                        "output": result.stdout[-1000:] if result.stdout else "",
                        "errors": result.stderr[-500:] if result.stderr else ""
                    }
                    
            except subprocess.TimeoutExpired:
                print("❌ ERROR: Simulación excedió tiempo límite")
                simulation_results = {
                    "status": "timeout",
                    "message": "La simulación excedió el tiempo límite de 10 minutos"
                }
            except Exception as sim_error:
                print(f"❌ ERROR en simulación: {str(sim_error)}")
                simulation_results = {
                    "status": "error",
                    "message": f"Error ejecutando simulación: {str(sim_error)}"
                }
        else:
            print(f"❌ Script de simulación no encontrado: {simulation_script}")
            simulation_results = {
                "status": "error",
                "message": f"Script de simulación no encontrado: {simulation_script}"
            }
        
        # Preparar respuesta
        response_data = {
            "status": "success",
            "modification_results": modification_results,
            "simulation_results": simulation_results,
            "csv_processing_results": csv_processing_results,
            "vhdl_file": vhdl_file_path,
            "backup_file": backup_path,
            "message": "Modificación de pesos, simulación y procesamiento CSV completado"
        }
        
        print("✅ Respuesta preparada exitosamente")
        return sanitize_for_json(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ ERROR general: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en el proceso: {str(e)}")


def modify_vhdl_weights_and_bias(content: str, fault_config: dict) -> str:
    """
    Modifica los pesos (FMAP_X) y bias (BIAS_VAL_X) en el contenido del archivo VHDL.
    
    Args:
        content: Contenido del archivo VHDL
        fault_config: Configuración de fallos con filter_faults y bias_faults
    
    Returns:
        Contenido modificado del archivo VHDL
    """
    import re
    
    modified_content = content
    
    # Procesar modificaciones de filtros (pesos)
    filter_faults = fault_config.get('filter_faults', [])
    if filter_faults:
        print(f"🔧 Modificando filtros: {filter_faults}")
        
        # Agrupar fallos por filter_name
        filter_groups = {}
        for fault in filter_faults:
            filter_name = fault.get('filter_name', '')
            if filter_name not in filter_groups:
                filter_groups[filter_name] = []
            
            # Convertir el formato del frontend al formato esperado
        position_config = {
            'row': fault.get('row', 0),
            'col': fault.get('col', 0),
            'bit_positions': [fault.get('bit_position', 0)],  # Convertir a lista
            'fault_type': fault.get('fault_type', 'bitflip')  # Incluir tipo de fallo
        }
        filter_groups[filter_name].append(position_config)
        
        # Procesar cada grupo de filtros
        for filter_name, positions in filter_groups.items():
            if positions:
                print(f"🔧 Procesando {filter_name} con posiciones: {positions}")
                modified_content = modify_filter_in_vhdl(modified_content, filter_name, positions)
    
    # Procesar modificaciones de bias
    bias_faults = fault_config.get('bias_faults', [])
    if bias_faults:
        print(f"🔧 Modificando bias: {bias_faults}")
        
        # Agrupar fallos por bias_name
        bias_groups = {}
        for fault in bias_faults:
            bias_name = fault.get('bias_name', '')
            if bias_name not in bias_groups:
                bias_groups[bias_name] = []
            
            # Agregar la posición del bit y el tipo de fallo
            fault_info = {
                'bit_position': fault.get('bit_position', 0),
                'fault_type': fault.get('fault_type', 'bitflip')
            }
            bias_groups[bias_name].append(fault_info)
        
        # Procesar cada grupo de bias
        for bias_name, fault_infos in bias_groups.items():
            if fault_infos:
                print(f"🔧 Procesando {bias_name} con fallos: {fault_infos}")
                modified_content = modify_bias_in_vhdl(modified_content, bias_name, fault_infos)
    
    return modified_content

def modify_filter_in_vhdl(content: str, filter_name: str, positions: list) -> str:
    """
    Modifica un filtro específico (FMAP_X) en el contenido VHDL.
    
    Args:
        content: Contenido del archivo VHDL
        filter_name: Nombre del filtro (ej: "FMAP_1")
        positions: Lista de posiciones a modificar [{"row": 0, "col": 1, "bit_positions": [0, 1]}]
    
    Returns:
        Contenido modificado
    """
    import re
    
    # Buscar la definición del filtro en el VHDL
    pattern = rf'constant {filter_name}: FILTER_TYPE:=\s*\((.*?)\);'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print(f"⚠️ No se encontró el filtro {filter_name}")
        return content
    
    filter_definition = match.group(1)
    print(f"✅ Encontrado filtro {filter_name}")
    
    # Parsear la matriz del filtro
    rows = []
    for line in filter_definition.split('\n'):
        line = line.strip()
        if line.startswith('(') and line.endswith('),') or line.endswith(')'):
            # Extraer valores entre paréntesis
            row_match = re.search(r'\((.*?)\)', line)
            if row_match:
                values = [v.strip().strip('"') for v in row_match.group(1).split(',')]
                rows.append(values)
    
    print(f"🔍 Matriz original del filtro {filter_name}: {len(rows)}x{len(rows[0]) if rows else 0}")
    
    # Aplicar modificaciones
    for pos_config in positions:
        row = pos_config.get('row', 0)
        col = pos_config.get('col', 0)
        bit_positions = pos_config.get('bit_positions', [])
        fault_type = pos_config.get('fault_type', 'bitflip')
        
        if 0 <= row < len(rows) and 0 <= col < len(rows[row]):
            original_value = rows[row][col]
            modified_value = apply_bit_faults(original_value, bit_positions, fault_type)
            rows[row][col] = modified_value
            print(f"✅ Modificado {filter_name}[{row}][{col}]: {original_value} → {modified_value} (tipo: {fault_type})")
        else:
            print(f"⚠️ Posición inválida {filter_name}[{row}][{col}]")
    
    # Reconstruir la definición del filtro
    new_filter_lines = []
    for i, row in enumerate(rows):
        formatted_values = ','.join([f'"{val}"' for val in row])
        if i == len(rows) - 1:
            new_filter_lines.append(f'\t\t({formatted_values})')
        else:
            new_filter_lines.append(f'\t\t({formatted_values}),')
    
    new_filter_definition = f'constant {filter_name}: FILTER_TYPE:= (\n' + '\n'.join(new_filter_lines) + '\n\t);'
    
    # Reemplazar en el contenido
    return content.replace(match.group(0), new_filter_definition)

def modify_bias_in_vhdl(content: str, bias_name: str, fault_infos: list) -> str:
    """
    Modifica un bias específico (BIAS_VAL_X) en el contenido VHDL.
    
    Args:
        content: Contenido del archivo VHDL
        bias_name: Nombre del bias (ej: "BIAS_VAL_1")
        fault_infos: Lista de información de fallos [{'bit_position': 0, 'fault_type': 'stuck_at_0'}, ...]
    
    Returns:
        Contenido modificado
    """
    import re
    
    # Buscar la definición del bias en el VHDL
    pattern = rf'constant {bias_name}: signed \(15 downto 0\) := "([01]+)";'
    match = re.search(pattern, content)
    
    if not match:
        print(f"⚠️ No se encontró el bias {bias_name}")
        return content
    
    original_value = match.group(1)
    modified_value = original_value
    
    # Aplicar cada fallo
    for fault_info in fault_infos:
        bit_position = fault_info.get('bit_position', 0)
        fault_type = fault_info.get('fault_type', 'bitflip')
        modified_value = apply_bit_faults(modified_value, [bit_position], fault_type)
    
    print(f"✅ Modificado {bias_name}: {original_value} → {modified_value}")
    
    # Reemplazar en el contenido
    new_definition = f'constant {bias_name}: signed (15 downto 0) := "{modified_value}";'
    return content.replace(match.group(0), new_definition)

def apply_bit_faults(binary_string: str, bit_positions: list, fault_type: str = 'bitflip') -> str:
    """
    Aplica fallos de bit según el tipo especificado a una cadena binaria.
    
    Args:
        binary_string: Cadena binaria original
        bit_positions: Lista de posiciones de bits a modificar
        fault_type: Tipo de fallo ('bitflip', 'stuck_at_0', 'stuck_at_1')
    
    Returns:
        Cadena binaria modificada
    """
    if not bit_positions:
        return binary_string
    
    # Convertir a lista para poder modificar
    bits = list(binary_string)
    
    for bit_pos in bit_positions:
        if 0 <= bit_pos < len(bits):
            original_bit = bits[bit_pos]
            
            if fault_type == 'stuck_at_0':
                # Forzar el bit a 0
                bits[bit_pos] = '0'
                print(f"  🔒 Bit {bit_pos}: {original_bit} → 0 (stuck-at-0)")
            elif fault_type == 'stuck_at_1':
                # Forzar el bit a 1
                bits[bit_pos] = '1'
                print(f"  🔒 Bit {bit_pos}: {original_bit} → 1 (stuck-at-1)")
            else:  # bitflip o cualquier otro tipo
                # Invertir el bit (comportamiento original)
                bits[bit_pos] = '1' if bits[bit_pos] == '0' else '0'
                print(f"  🔄 Bit {bit_pos}: {original_bit} → {bits[bit_pos]} (bit-flip)")
        else:
            print(f"  ⚠️ Posición de bit inválida: {bit_pos} (longitud: {len(bits)})")
    
    return ''.join(bits)

@app.post("/modify_vhdl_weights/")
async def modify_vhdl_weights(
    filter_faults: str = Body(None),
    bias_faults: str = Body(None),
    current_user: dict = Depends(get_current_user)
):
    """
    Modifica los pesos y bias en el archivo VHDL según la configuración de fallos.
    """
    try:
        print(f"🔍 DEBUG: Recibido filter_faults: {filter_faults}")
        print(f"🔍 DEBUG: Recibido bias_faults: {bias_faults}")
        
        # Ruta del archivo VHDL
        vhdl_file_path = "/home/davidgonzalez/Documentos/David_2025/4_CONV1_SAB_STUCKAT_DEC_RAM_TB/CONV1_SAB_STUCKAT_DEC_RAM.srcs/sources_1/new/CONV1_SAB_STUCK_DECOS.vhd"
        
        # Verificar que el archivo existe
        if not os.path.exists(vhdl_file_path):
            raise HTTPException(status_code=404, detail=f"Archivo VHDL no encontrado: {vhdl_file_path}")
        
        # Parsear configuraciones de fallos
        try:
            print(f"🔍 DEBUG: Parseando filter_faults: {filter_faults}")
            filter_config = json.loads(filter_faults) if filter_faults else {}
            print(f"🔍 DEBUG: Parseando bias_faults: {bias_faults}")
            bias_config = json.loads(bias_faults) if bias_faults else {}
            print(f"✅ Configuraciones parseadas - filter: {filter_config}, bias: {bias_config}")
        except json.JSONDecodeError as e:
            print(f"❌ ERROR: Error parseando JSON: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error parseando configuración JSON: {str(e)}")
        
        # Crear directorio temporal para respaldo
        temp_dir = f"/tmp/vhdl_backup_{uuid.uuid4()}"
        os.makedirs(temp_dir, exist_ok=True)
        print(f"✅ Directorio temporal creado: {temp_dir}")
        
        # Crear respaldo del archivo original
        backup_path = os.path.join(temp_dir, "CONV1_SAB_STUCK_DECOS_backup.vhd")
        shutil.copy2(vhdl_file_path, backup_path)
        print(f"✅ Respaldo creado: {backup_path}")
        
        # Modificar pesos en el archivo VHDL
        modification_results = {}
        try:
            print("🔄 Iniciando modificación de pesos...")
            # Leer el archivo VHDL
            with open(vhdl_file_path, 'r') as file:
                content = file.read()
            
            # Crear configuración combinada
            fault_config = {
                'filter_faults': filter_config,
                'bias_faults': bias_config
            }
            
            # Implementar modificación real de pesos y bias en el archivo VHDL
            modified_content = modify_vhdl_weights_and_bias(content, fault_config)
            
            # Escribir archivo modificado
            with open(vhdl_file_path, 'w') as file:
                file.write(modified_content)
            
            modification_results = {
                "status": "success",
                "filter_faults_applied": filter_config,
                "bias_faults_applied": bias_config,
                "backup_created": backup_path
            }
            print("✅ Modificación de pesos completada")
            
        except Exception as mod_error:
            print(f"❌ ERROR en modificación: {str(mod_error)}")
            # Restaurar archivo original en caso de error
            shutil.copy2(backup_path, vhdl_file_path)
            raise HTTPException(status_code=500, detail=f"Error modificando archivo VHDL: {str(mod_error)}")
        
        # Ejecutar script de simulación
        simulation_script = "/home/davidgonzalez/Documentos/David_2025/4_CONV1_SAB_STUCKAT_DEC_RAM_TB/CONV1_SAB_STUCKAT_DEC_RAM.sim/sim_1/behav/xsim/simulate.sh"
        simulation_results = {}
        print(f"🔍 DEBUG: Verificando script de simulación: {simulation_script}")
        
        if os.path.exists(simulation_script):
            print("✅ Script de simulación encontrado, ejecutando...")
            try:
                # Ejecutar el script de simulación con timeout de 10 minutos
                simulation_process = subprocess.run(
                    ["bash", simulation_script],
                    cwd=os.path.dirname(simulation_script),
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minutos
                )
                
                simulation_results = {
                    "status": "completed",
                    "return_code": simulation_process.returncode,
                    "stdout": simulation_process.stdout[:1000] if simulation_process.stdout else "",  # Limitar output
                    "stderr": simulation_process.stderr[:1000] if simulation_process.stderr else "",  # Limitar output
                    "script_path": simulation_script
                }
                print(f"✅ Simulación completada con código: {simulation_process.returncode}")
                
                # Buscar archivos Excel y CSV generados
                result_files = []
                simulation_dir = os.path.dirname(simulation_script)
                for root, dirs, files in os.walk(simulation_dir):
                    for file in files:
                        if file.endswith(('.xlsx', '.xls', '.csv')):
                            result_files.append(os.path.join(root, file))
                
                simulation_results["result_files"] = result_files
                simulation_results["excel_files"] = [f for f in result_files if f.endswith(('.xlsx', '.xls'))]  # Mantener compatibilidad
                simulation_results["csv_files"] = [f for f in result_files if f.endswith('.csv')]
                print(f"📊 Archivos de resultados encontrados: {len(result_files)} (Excel: {len(simulation_results['excel_files'])}, CSV: {len(simulation_results['csv_files'])})")
                
                # Procesar archivos CSV después de la simulación exitosa
                csv_processing_results = {}
                csv_files = [f for f in result_files if f.endswith('.csv') and 'Conv1' in f]
                
                if csv_files:
                    print(f"✅ Encontrados {len(csv_files)} archivos CSV para procesar")
                    # Importar el procesador CSV
                    from vhdl_hardware.csv_processor import CSVProcessor
                    
                    csv_processor = CSVProcessor()
                    processed_results = []
                    
                    for csv_file in csv_files:
                        try:
                            print(f"🔄 Procesando archivo CSV: {csv_file}")
                            result = csv_processor.process_simulation_csv(csv_file)
                            processed_results.append({
                                "file": csv_file,
                                "result": result
                            })
                            print(f"✅ Archivo CSV procesado exitosamente: {csv_file}")
                        except Exception as csv_error:
                            print(f"❌ Error procesando CSV {csv_file}: {str(csv_error)}")
                            processed_results.append({
                                "file": csv_file,
                                "error": str(csv_error)
                            })
                    
                    csv_processing_results = {
                        "status": "success",
                        "processed_files": len(processed_results),
                        "results": processed_results
                    }
                    simulation_results["csv_processing_results"] = csv_processing_results
                else:
                    print("⚠️ No se encontraron archivos CSV para procesar")
                    simulation_results["csv_processing_results"] = {
                        "status": "warning",
                        "message": "No se encontraron archivos CSV para procesar"
                    }
                
            except subprocess.TimeoutExpired:
                print("⏰ Simulación excedió tiempo límite")
                simulation_results = {
                    "status": "timeout",
                    "message": "La simulación excedió el tiempo límite de 10 minutos"
                }
            except Exception as sim_error:
                print(f"❌ ERROR en simulación: {str(sim_error)}")
                simulation_results = {
                    "status": "error",
                    "message": f"Error ejecutando simulación: {str(sim_error)}"
                }
        else:
            print(f"❌ Script de simulación no encontrado: {simulation_script}")
            simulation_results = {
                "status": "error",
                "message": f"Script de simulación no encontrado: {simulation_script}"
            }
        
        # Preparar respuesta
        response_data = {
            "status": "success",
            "modification_results": modification_results,
            "simulation_results": simulation_results,
            "vhdl_file": vhdl_file_path,
            "backup_file": backup_path,
            "message": "Modificación de pesos y simulación completada"
        }
        
        print("✅ Respuesta preparada exitosamente")
        return sanitize_for_json(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ ERROR general: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en el proceso: {str(e)}")

if __name__ == "__main__":
    # Configuración del servidor
    host = "0.0.0.0"  # Esto permite conexiones desde cualquier IP
    port = int(os.getenv("PORT", 8001))  # Usar variable de entorno PORT para despliegue en la nube

    print(f"Iniciando servidor en http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)