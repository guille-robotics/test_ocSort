import torch

# --- RUTAS (se sobreescriben en batch_test.py) ---
VIDEO_IN  = "videos_para_testear/video3.mp4"
VIDEO_OUT = "videos_salida/v3_resultado_video3.mp4"
CSV_OUT   = "videos_salida/v3_resultado_video3.csv"
CROPS_DIR = "recortes_ropa_final_osnet17"

# --- MODELOS ---
DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'
DETECTOR_WEIGHTS = 'weights/detector/FAL-zi_v1_DB-egana-v2_best.pt'
REID_WEIGHTS     = 'osnet_x1_0_msmt17.pt'

# --- RENDIMIENTO GPU ---
# Activar fp16 (half precision) en GPU para mayor velocidad.
# RTX 2070 / RTX 5090 lo soportan. En CPU siempre se ignora.
USE_FP16 = (DEVICE == 'cuda')

# --- PARÁMETROS DE VISIÓN ---
TARGET_CLASSES  = [0]
DET_CONFIDENCE  = 0.25    # subido ligeramente para filtrar falsas detecciones
NMS_IOU         = 0.35    # NMS manual antes de enviar al tracker
MIN_AREA        = 25000   # píxeles² mínimos para aceptar una detección

# --- PARÁMETROS STRONGSORT ---
# Reducir max_age y aumentar min_hits ayuda a tener menos IDs espurios.
TRACK_MAX_AGE      = 45    # ↓ (era 60) → tracks perdidos expiran antes
TRACK_MIN_HITS     = 3     # ↑ (era 2)  → un track necesita 3 frames para confirmarse
TRACK_MAX_DIST     = 0.20  # ↓ (era 0.25) → más estricto en similitud de apariencia
TRACK_MAX_IOU_DIST = 0.65  # ↓ (era 0.70) → más estricto en asociación espacial
TRACK_EMA_ALPHA    = 0.90  # suavizado EMA (memoria visual larga)
TRACK_MC_LAMBDA    = 0.995 # peso de momentum en la predicción de Kalman
TRACK_NN_BUDGET    = 150   # ↑ más embeddings almacenados para mejor Re-ID

# --- PARÁMETROS RECUPERACIÓN DE ID HÍBRIDA ---
ID_RECOVERY_MAX_DIST          = 220   # ↓ px máx para match espacial (menos falsos)
ID_RECOVERY_MAX_AGE           = 45    # ↓ frames máx perdido (coherente con max_age)
ID_RECOVERY_SCORE_THRESHOLD   = 0.38  # ↑ umbral más exigente para recuperar un ID
ID_RECOVERY_SPATIAL_WEIGHT    = 0.30  # peso de la distancia espacial
ID_RECOVERY_APPEARANCE_WEIGHT = 0.70  # peso del color HSV (más relevante)

# --- PARÁMETROS ESTABILIDAD DE ID ---
# Un ID solo se "confirma" tras este número de frames consecutivos detectados.
# IDs que duran menos frames son tracks de ruido y no se cuentan en el resumen.
TRACK_MIN_STABLE_FRAMES = 5   # nuevo: filtrar tracks efímeros (ruido de 1-3 frames)

# --- PARÁMETROS LÓGICA DE ZONAS ---
ZONE_WARMUP_FRAMES   = 15    # ↓ (era 20) frames de gracia antes de evaluar zonas
ZONE_ALERT_FRAMES    = 20    # frames en zona bolsa sin escanear → alerta
OCCLUSION_IOU_THRESH = 0.30  # IoU entre prendas para considerarlas en oclusión

# ==========================================
# ZONAS DE INTERÉS (ROI) — fallback
# Se usan solo si el video no tiene ROI definida en config.yaml
# ==========================================
ZONA_ESCANER = [(642, 12), (645, 433), (1098, 420), (1099, 14)]
ZONA_BOLSA   = [(1172, 20), (1184, 403), (1475, 403), (1466, 40)]