from fastapi import FastAPI, HTTPException, Body, Depends, status, UploadFile, File
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

# Importar desde nuestros módulos
from models import ImageToVHDLRequest, FaultInjectorRequest, FaultInjectorInferenceRequest, FaultInjectionConfig

# Importar la función de cuantización
from model_quantization import modify_and_save_weights

# Importar bibliotecas adicionales si no están ya importadas
import keras

# Importar inferencia manual
from fault_injection.manual_inference import ManualInference

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
def register(user: UserCreate):
    """Registrar nuevo usuario"""
    try:
        new_user = create_user(user)
        return {
            "success": True,
            "message": "Usuario creado exitosamente",
            "user": new_user
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
def get_models():
    """Endpoint para obtener la lista de modelos disponibles"""
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
def list_models():
    """Endpoint legacy para compatibilidad"""
    return get_models()

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

if __name__ == "__main__":
    # Configuración del servidor
    host = "0.0.0.0"  # Esto permite conexiones desde cualquier IP
    port = int(os.getenv("PORT", 8001))  # Usar variable de entorno PORT para despliegue en la nube

    print(f"Iniciando servidor en http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)