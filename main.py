from fastapi import FastAPI, HTTPException, Body, Depends, status
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

# Importar desde nuestros módulos
from models import CNNConfig, ImageToVHDLRequest
from utils import create_cnn, generate_model_filename

# Importar la función de cuantización
from model_quantization import modify_and_save_weights

# Importar bibliotecas adicionales si no están ya importadas
import keras

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

@app.post("/create_cnn/")
def create_cnn_endpoint(config: CNNConfig, current_user: dict = Depends(get_current_user)):
    try:
        model = create_cnn(config)
        model.summary() # para corroborar la arquitectura del modelo
        
        # Generar nombre de archivo para el modelo
        model_filename = generate_model_filename(config.model_name)
        
        # Guardar el modelo en disco
        model.save(model_filename)
        
        return {
            "message": "Modelo creado con éxito", 
            "layers": [layer.name for layer in model.layers],
            "model_path": model_filename
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

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

@app.get("/list_models/")
def list_models():
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
async def convert_image_to_vhdl(request: ImageToVHDLRequest):
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
        
        # Guardar el código VHDL en un archivo de texto con la estructura de Memoria_Imagen.vhd
        # output_path = "c:/Users/PC/Documents/UNIVERSIDAD/Project/Back_project/Memoria_Imagen.vhdl.txt"
        output_path = os.path.join(os.path.dirname(__file__), "Memoria_Imagen.vhdl.txt")
        with open(output_path, "w") as f:
            f.write(vhdl_code)
        
        return {
            "success": True,
            "decimal_matrix": decimal_matrix,
            "hex_matrix": hex_matrix,
            "vhdl_code": vhdl_code,
            "file_path": output_path
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
def extract_model_weights(request: ModelWeightsRequest):
    try:
        if not os.path.exists(request.model_path):
            raise HTTPException(status_code=404, detail=f"Modelo no encontrado: {os.path.basename(request.model_path)}")
        
        # Crear directorio de salida
        output_dir = request.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
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
        
        return {
            "success": True,
            "message": f"Pesos y sesgos extraídos exitosamente con {bits_value} bits",
            "model": os.path.basename(request.model_path),
            "output_dir": output_dir,
            "files": generated_files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint para descargar archivos generados
@app.get("/download_file/")
def download_file(file_path: str):
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {os.path.basename(file_path)}")
        
        # Leer el contenido del archivo
        with open(file_path, "r") as f:
            content = f.read()
        
        return {"content": content, "filename": os.path.basename(file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Configuración del servidor
    host = "0.0.0.0"  # Esto permite conexiones desde cualquier IP
    port = int(os.getenv("PORT", 8000))  # Usar variable de entorno PORT para despliegue en la nube
    
    print(f"Iniciando servidor en http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)