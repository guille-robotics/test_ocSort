import torch
from pathlib import Path

# --- RUTAS ---
VIDEO_IN  = "videos_para_testear/video3.mp4"
VIDEO_OUT = "videos_salida/v4_resultado_video3.mp4"
CSV_OUT   = "videos_salida/v4_resultado_video3.csv"
CROPS_DIR = "recortes_ropa_final_osnet17"

# --- MODELOS ---
DEVICE           = 'cuda' if torch.cuda.is_available() else 'cpu'
DETECTOR_WEIGHTS = 'weights/detector/FAL-zi_v1_DB-egana-v2_best.pt'
REID_WEIGHTS     = 'osnet_x1_0_msmt17.pt'

# --- RENDIMIENTO GPU ---
USE_FP16 = (DEVICE == 'cuda')   # half-precision en GPU

# ══════════════════════════════════════════════════════════════════════════════
# TRACKER
# Opciones: 'botsort' | 'strongsort' | 'bytetrack'
# botsort   → combina IoU + ReID + Kalman; mejor trade-off calidad/velocidad
# strongsort → apariencia fuerte, más lento
# bytetrack → sin apariencia, más rápido, menos re-ID
# ══════════════════════════════════════════════════════════════════════════════
TRACKER_TYPE = 'botsort'

# ── BoT-SORT ──────────────────────────────────────────────────────────────────
BOTSORT_TRACK_HIGH_THRESH = 0.50   # conf mínima para 1ª ronda de asociación
BOTSORT_NEW_TRACK_THRESH  = 0.60   # conf mínima para crear track nuevo
BOTSORT_TRACK_BUFFER      = 50     # frames que un track puede estar perdido
BOTSORT_MATCH_THRESH      = 0.65   # umbral IoU para asociación
BOTSORT_PROXIMITY_THRESH  = 0.50   # umbral de proximidad espacial
BOTSORT_APPEARANCE_THRESH = 0.28   # umbral de distancia de apariencia ReID

# ── ByteTrack (fallback sin ReID) ─────────────────────────────────────────────
BYTE_MIN_CONF    = 0.25
BYTE_TRACK_BUFF  = 50
BYTE_MATCH_THRESH = 0.80

# ── StrongSORT ────────────────────────────────────────────────────────────────
TRACK_MAX_AGE      = 50
TRACK_MIN_HITS     = 3
TRACK_MAX_DIST     = 0.20
TRACK_MAX_IOU_DIST = 0.65
TRACK_EMA_ALPHA    = 0.90
TRACK_MC_LAMBDA    = 0.995
TRACK_NN_BUDGET    = 150

# --- PARÁMETROS DETECCIÓN ---
TARGET_CLASSES  = [0]
DET_CONFIDENCE  = 0.25
NMS_IOU         = 0.35
MIN_AREA        = 25000

# --- RECUPERACIÓN DE ID HÍBRIDA ---
ID_RECOVERY_MAX_DIST          = 220
ID_RECOVERY_MAX_AGE           = 50
ID_RECOVERY_SCORE_THRESHOLD   = 0.38
ID_RECOVERY_SPATIAL_WEIGHT    = 0.30
ID_RECOVERY_APPEARANCE_WEIGHT = 0.70

# --- ESTABILIDAD DE ID ---
# Un ID se "confirma" solo tras este número de frames consecutivos
TRACK_MIN_STABLE_FRAMES = 5

# --- ZONAS ---
ZONE_WARMUP_FRAMES   = 15
ZONE_ALERT_FRAMES    = 20
OCCLUSION_IOU_THRESH = 0.30

# ══════════════════════════════════════════════════════════════════════════════
# CONSOLIDACIÓN DE IDs → CUENTA REAL DE PRENDAS
# Post-proceso que fusiona IDs fragmentados en "prendas reales"
# ══════════════════════════════════════════════════════════════════════════════
CONSOLIDATION_MAX_GAP    = 90    # frames máx de brecha temporal para fusionar
CONSOLIDATION_SIM_THRESH = 0.68  # similitud mínima de histograma HSV para fusionar

# --- ROI FALLBACK ---
ZONA_ESCANER = [(642, 12), (645, 433), (1098, 420), (1099, 14)]
ZONA_BOLSA   = [(1172, 20), (1184, 403), (1475, 403), (1466, 40)]