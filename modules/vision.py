from pathlib import Path
from ultralytics import RTDETR
from boxmot import OcSort
import config

def inicializar_modelos():
    print(f"--- Cargando RT-DETR en {config.DEVICE} ---")
    model = RTDETR(config.DETECTOR_WEIGHTS)
    model.to(config.DEVICE)

    print(f"--- Cargando OC-SORT + ReID ---")
    tracker = OcSort(
        model=Path(config.REID_WEIGHTS),
        device=config.DEVICE,
        per_class=True, 
        det_thresh=config.DET_CONFIDENCE,
        max_age=config.TRACK_MAX_AGE,
        min_hits=config.TRACK_MIN_HITS,
        iou_threshold=config.TRACK_IOU,
        asso_func=config.TRACK_ASSO_FUNC,
        delta_t=config.TRACK_DELTA_T,
        inertia=config.TRACK_INERTIA,
        use_byte=True
    )
    
    return model, tracker