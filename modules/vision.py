from pathlib import Path
import torch
from ultralytics import RTDETR
from boxmot import StrongSort
import config


def inicializar_modelos():
    print(f"--- Cargando RT-DETR en {config.DEVICE} ---")
    model = RTDETR(config.DETECTOR_WEIGHTS)
    model.to(config.DEVICE)

    # StrongSORT necesita un torch.device explícito (no string 'cuda')
    if config.DEVICE == 'cuda':
        reid_device = torch.device('cuda:0')
    else:
        reid_device = torch.device('cpu')

    print(f"--- Cargando StrongSORT + ReID (OSNet) en {reid_device} ---")
    tracker = StrongSort(
        reid_weights=Path(config.REID_WEIGHTS),
        device=reid_device,
        half=False,             # fp16 desactivado — activar si reid_weights es .engine o .onnx
        # StrongSORT-specific
        min_conf=config.DET_CONFIDENCE,
        max_cos_dist=config.TRACK_MAX_DIST,
        max_iou_dist=config.TRACK_MAX_IOU_DIST,
        n_init=config.TRACK_MIN_HITS,
        nn_budget=config.TRACK_NN_BUDGET,
        mc_lambda=config.TRACK_MC_LAMBDA,
        ema_alpha=config.TRACK_EMA_ALPHA,
        # BaseTracker (via **kwargs)
        max_age=config.TRACK_MAX_AGE,
    )

    return model, tracker