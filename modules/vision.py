from pathlib import Path
import torch
from ultralytics import RTDETR
import config


def inicializar_modelos():
    """
    Carga RT-DETR + el tracker seleccionado via config.TRACKER_TYPE.

    Trackers soportados:
      'botsort'    — BoT-SORT: IoU + ReID + Kalman mejorado (RECOMENDADO)
      'strongsort' — StrongSORT: apariencia fuerte, más lento
      'bytetrack'  — ByteTrack: IoU puro, sin ReID, muy rápido
    """
    print(f"--- Cargando RT-DETR en {config.DEVICE} ---")
    model = RTDETR(config.DETECTOR_WEIGHTS)
    model.to(config.DEVICE)

    device = (torch.device('cuda:0')
              if config.DEVICE == 'cuda'
              else torch.device('cpu'))

    tracker_name = config.TRACKER_TYPE.lower()

    # ── BoT-SORT ──────────────────────────────────────────────────────────────
    if tracker_name == 'botsort':
        from boxmot import BotSort
        print(f"--- Cargando BoT-SORT + ReID (OSNet) en {device} ---")
        tracker = BotSort(
            reid_weights=Path(config.REID_WEIGHTS),
            device=device,
            half=config.USE_FP16,
            track_high_thresh=config.BOTSORT_TRACK_HIGH_THRESH,
            new_track_thresh=config.BOTSORT_NEW_TRACK_THRESH,
            track_buffer=config.BOTSORT_TRACK_BUFFER,
            match_thresh=config.BOTSORT_MATCH_THRESH,
            proximity_thresh=config.BOTSORT_PROXIMITY_THRESH,
            appearance_thresh=config.BOTSORT_APPEARANCE_THRESH,
            with_reid=True,
        )

    # ── ByteTrack ─────────────────────────────────────────────────────────────
    elif tracker_name == 'bytetrack':
        from boxmot import ByteTrack
        print(f"--- Cargando ByteTrack (sin ReID) en {device} ---")
        tracker = ByteTrack(
            min_conf=config.BYTE_MIN_CONF,
            track_buffer=config.BYTE_TRACK_BUFF,
            match_thresh=config.BYTE_MATCH_THRESH,
        )

    # ── StrongSORT (legacy) ───────────────────────────────────────────────────
    elif tracker_name == 'strongsort':
        from boxmot import StrongSort
        print(f"--- Cargando StrongSORT + ReID (OSNet) en {device} ---")
        tracker = StrongSort(
            reid_weights=Path(config.REID_WEIGHTS),
            device=device,
            half=False,
            min_conf=config.DET_CONFIDENCE,
            max_cos_dist=config.TRACK_MAX_DIST,
            max_iou_dist=config.TRACK_MAX_IOU_DIST,
            n_init=config.TRACK_MIN_HITS,
            nn_budget=config.TRACK_NN_BUDGET,
            mc_lambda=config.TRACK_MC_LAMBDA,
            ema_alpha=config.TRACK_EMA_ALPHA,
            max_age=config.TRACK_MAX_AGE,
        )

    else:
        raise ValueError(f"TRACKER_TYPE desconocido: '{config.TRACKER_TYPE}'. "
                         f"Usa 'botsort', 'bytetrack' o 'strongsort'.")

    print(f"    Tracker: {tracker_name.upper()} listo.")
    return model, tracker