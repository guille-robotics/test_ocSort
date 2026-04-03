import torch

# --- RUTAS ---
VIDEO_IN = "videos_para_testear/video3.mp4"
VIDEO_OUT = "videos_salida/TEST_giou_recovery_video3.mp4"
CSV_OUT = "videos_salida/TEST_giou_recovery_video3.csv"
CROPS_DIR = "recortes_ropa_final_osnet17"

# --- MODELOS ---
DEVICE = 'cuda' # Forzado a usar GPU (RTX 2070). Si da error, falta el PyTorch con CUDA.
DETECTOR_WEIGHTS = 'weights/detector/FAL-zi_v1_DB-egana-v2_best.pt'
REID_WEIGHTS = 'osnet_x1_0_msmt17.pt' 

# --- PARÁMETROS DE VISIÓN ---
TARGET_CLASSES = [0]  
DET_CONFIDENCE = 0.2

# --- PARÁMETROS OC-SORT ---
# GIoU como función de asociación: funciona incluso cuando las cajas NO se superponen.
# Esto es clave para movimientos bruscos donde el IoU cae a 0 y el tracker pierde la pista.
TRACK_MAX_AGE = 60        # Mantener el track vivo max 2 segundos sin detección (a 30fps)
TRACK_MIN_HITS = 2        # 2 detecciones para confirmar una prenda nueva
TRACK_IOU = 0.20          # Umbral mínimo de asociación
TRACK_ASSO_FUNC = 'giou'  # GIoU tolera desplazamientos bruscos (NO usar 'iou' puro)
TRACK_DELTA_T = 5         # Ventana de velocidad del filtro de Kalman (más alta = más tolerante a saltos)
TRACK_INERTIA = 0.4       # Inercia del predictor (más alta = prioriza predicción ante saltos)

# --- PARÁMETROS RECUPERACIÓN DE ID ---
# Distancia máxima (en píxeles) para considerar que un ID nuevo es el mismo que uno perdido.
# Para una cámara cenital a esa altura, 200px es un salto brusco pero plausible.
ID_RECOVERY_MAX_DIST = 200   # píxeles
ID_RECOVERY_MAX_AGE  = 45    # frames que puede estar perdido un track y aun así recuperarse

# ==========================================
# NUEVO: ZONAS DE INTERÉS (ROI)
# Coordenadas (X, Y) para dibujar los polígonos. 
# AJUSTA ESTOS VALORES según la vista de tu cámara.
# ==========================================
# ==========================================
# ZONAS DE INTERÉS (ROI)
# ==========================================
ZONA_ESCANER = [(642, 12), (645, 433), (1098, 420), (1099, 14)]
ZONA_BOLSA = [(1172, 20), (1184, 403), (1475, 403), (1466, 40)]