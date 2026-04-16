import os
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = "tu_clave_secreta_super_segura_cambiala_en_produccion"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

USERS_COLLECTION = "users"
FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LAYER_OUTPUTS_DIR = os.path.join(BASE_DIR, "layer_outputs")
VHDL_OUTPUTS_DIR = os.path.join(BASE_DIR, "vhdl_outputs")
MODEL_WEIGHTS_OUTPUTS_DIR = os.path.join(BASE_DIR, "model_weights_outputs")

VHDL_FILE_PATH = "/home/davidgonzalez/Documentos/David_2025/4_CONV1_SAB_STUCKAT_DEC_RAM_TB/CONV1_SAB_STUCKAT_DEC_RAM.srcs/sources_1/new/CONV1_SAB_STUCK_DECOS.vhd"
VHDL_SIM_DIR = "/home/davidgonzalez/Documentos/David_2025/4_CONV1_SAB_STUCKAT_DEC_RAM_TB/CONV1_SAB_STUCKAT_DEC_RAM.sim/sim_1/behav/xsim"
