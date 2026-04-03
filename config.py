import torch

# --- RUTAS (se sobreescriben en batch_test.py) ---
VIDEO_IN  = "videos_para_testear/video3.mp4"
VIDEO_OUT = "videos_salida/v2_resultado_video3.mp4"
CSV_OUT   = "videos_salida/v2_resultado_video3.csv"
CROPS_DIR = "recortes_ropa_final_osnet17"

# --- MODELOS ---
DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'
DETECTOR_WEIGHTS = 'weights/detector/FAL-zi_v1_DB-egana-v2_best.pt'
REID_WEIGHTS     = 'osnet_x1_0_msmt17.pt'

# --- PARÁMETROS DE VISIÓN ---
TARGET_CLASSES  = [0]
DET_CONFIDENCE  = 0.2
NMS_IOU         = 0.3    # NMS manual antes de enviar al tracker
MIN_AREA        = 25000  # píxeles² mínimos para aceptar una detección

# --- PARÁMETROS STRONGSORT ---
# StrongSORT usa EMA de embeddings de apariencia → memoria visual tras oclusiones.
TRACK_MAX_AGE      = 60    # frames que un track puede estar perdido antes de descartarse
TRACK_MIN_HITS     = 2     # detecciones consecutivas para confirmar un track nuevo
TRACK_MAX_DIST     = 0.25  # distancia coseno máxima para match de apariencia (0=idéntico)
TRACK_MAX_IOU_DIST = 0.70  # distancia IoU máxima para match espacial
TRACK_EMA_ALPHA    = 0.90  # suavizado EMA de features (más alto = memoria más larga)
TRACK_MC_LAMBDA    = 0.995 # peso de momentum en la predicción
TRACK_NN_BUDGET    = 100   # máximo de embeddings almacenados por track

# --- PARÁMETROS RECUPERACIÓN DE ID HÍBRIDA ---
# Combina distancia espacial + similitud de histograma de color HSV.
ID_RECOVERY_MAX_DIST          = 250   # px máx para buscar un match espacial
ID_RECOVERY_MAX_AGE           = 60    # frames máx que puede llevar perdido un track
ID_RECOVERY_SCORE_THRESHOLD   = 0.30  # score mínimo combinado para recuperar un ID
ID_RECOVERY_SPATIAL_WEIGHT    = 0.35  # peso de la distancia espacial en el score
ID_RECOVERY_APPEARANCE_WEIGHT = 0.65  # peso de la similitud de color en el score

# --- PARÁMETROS LÓGICA DE ZONAS ---
ZONE_WARMUP_FRAMES   = 20   # Frames de gracia antes de aplicar lógica de alerta a un ID nuevo
ZONE_ALERT_FRAMES    = 20   # Frames consecutivos en zona bolsa sin escanear para disparar alerta
OCCLUSION_IOU_THRESH = 0.30 # IoU entre dos prendas para considerarlas en oclusión

# ==========================================
# ZONAS DE INTERÉS (ROI)
# ==========================================
ZONA_ESCANER = [(642, 12), (645, 433), (1098, 420), (1099, 14)]
ZONA_BOLSA   = [(1172, 20), (1184, 403), (1475, 403), (1466, 40)]