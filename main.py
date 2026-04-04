import cv2
import os
import numpy as np
import math
import csv
import yaml
import config
from pathlib import Path
from modules.vision import inicializar_modelos
from modules.ui import crear_lienzo, dibujar_cajas, dibujar_panel


# ==============================================================================
# UTILIDADES
# ==============================================================================

def calcular_iou(boxA, boxB):
    """IoU entre dos cajas [x1,y1,x2,y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)


def cargar_roi_para_video(video_path: str, yaml_path: str = "config.yaml"):
    """
    Carga los polígonos de ROI del YAML para el video dado.
    Si no hay entrada en el YAML usa los valores de config.py.
    """
    clave = Path(video_path).stem  # e.g. "video3"
    rois  = {}

    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            rois = yaml.safe_load(f) or {}

    if clave in rois:
        escaner = np.array(rois[clave]["Zona_Escaner"], np.int32)
        bolsa   = np.array(rois[clave]["Zona_Bolsa"],   np.int32)
        print(f"[ROI] '{clave}' → usando ROI del YAML")
    else:
        escaner = np.array(config.ZONA_ESCANER, np.int32)
        bolsa   = np.array(config.ZONA_BOLSA,   np.int32)
        print(f"[ROI] '{clave}' → usando ROI de config.py (fallback)")

    return escaner, bolsa


# ==============================================================================
# SISTEMA DE RECUPERACIÓN DE ID HÍBRIDO
# ==============================================================================

class IdRecoverySystem:
    """
    Mantiene IDs estables, recuperando el ID original cuando StrongSORT
    crea uno nuevo tras una oclusión.

    v3: agrega TRACK_MIN_STABLE_FRAMES para filtrar tracks de ruido efímeros.
    """
    def __init__(self):
        self.id_map        = {}
        self.id_inverso    = {}
        self.ultima_pos    = {}
        self.ultimo_frame  = {}
        self.histograma    = {}
        self.frames_vistos = {}
        self.confirmado    = {}
        self.siguiente_id  = 1

    def _extraer_histograma(self, frame, x1, y1, x2, y2):
        ix1 = max(0, int(x1));  iy1 = max(0, int(y1))
        ix2 = min(frame.shape[1], int(x2)); iy2 = min(frame.shape[0], int(y2))
        crop = frame[iy1:iy2, ix1:ix2]
        if crop.size == 0:
            return None
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    def _similitud_apariencia(self, hist_a, hist_b):
        if hist_a is None or hist_b is None:
            return 0.5
        return float(max(0.0, cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)))

    def _actualizar_histograma(self, id_estable, nuevo_hist):
        if nuevo_hist is None:
            return
        alpha = config.TRACK_EMA_ALPHA
        if id_estable in self.histograma and self.histograma[id_estable] is not None:
            self.histograma[id_estable] = (alpha * self.histograma[id_estable]
                                           + (1 - alpha) * nuevo_hist)
        else:
            self.histograma[id_estable] = nuevo_hist.copy()

    def get_id_estable(self, tracker_id, cx, cy, frame_count, frame, bbox):
        if tracker_id in self.id_map:
            id_estable = self.id_map[tracker_id]
            self.ultima_pos[id_estable]    = (cx, cy)
            self.ultimo_frame[id_estable]  = frame_count
            self.frames_vistos[id_estable] = self.frames_vistos.get(id_estable, 0) + 1
            if self.frames_vistos[id_estable] >= config.TRACK_MIN_STABLE_FRAMES:
                self.confirmado[id_estable] = True
            nuevo_hist = self._extraer_histograma(frame, *bbox)
            self._actualizar_histograma(id_estable, nuevo_hist)
            return id_estable

        hist_nuevo  = self._extraer_histograma(frame, *bbox)
        mejor_id    = None
        mejor_score = -float('inf')

        for id_est, (ox, oy) in self.ultima_pos.items():
            if id_est in self.id_inverso and self.id_inverso[id_est] in self.id_map:
                continue
            if frame_count - self.ultimo_frame.get(id_est, 0) > config.ID_RECOVERY_MAX_AGE:
                continue
            dist = math.hypot(cx - ox, cy - oy)
            if dist > config.ID_RECOVERY_MAX_DIST:
                continue

            score_dist = 1.0 - (dist / config.ID_RECOVERY_MAX_DIST)
            score_apar = self._similitud_apariencia(
                hist_nuevo, self.histograma.get(id_est))
            score_total = (config.ID_RECOVERY_SPATIAL_WEIGHT    * score_dist
                         + config.ID_RECOVERY_APPEARANCE_WEIGHT * score_apar)

            if score_total > mejor_score:
                mejor_score = score_total
                mejor_id    = id_est

        if mejor_id is not None and mejor_score >= config.ID_RECOVERY_SCORE_THRESHOLD:
            id_estable = mejor_id
        else:
            id_estable = self.siguiente_id
            self.siguiente_id += 1

        self.id_map[tracker_id]       = id_estable
        self.id_inverso[id_estable]   = tracker_id
        self.ultima_pos[id_estable]   = (cx, cy)
        self.ultimo_frame[id_estable] = frame_count
        self.frames_vistos[id_estable]= self.frames_vistos.get(id_estable, 0) + 1
        if self.frames_vistos[id_estable] >= config.TRACK_MIN_STABLE_FRAMES:
            self.confirmado[id_estable] = True
        self._actualizar_histograma(id_estable, hist_nuevo)
        return id_estable

    def get_ids_estables_activos(self):
        return set(self.id_map.values())


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    os.makedirs(config.CROPS_DIR, exist_ok=True)
    model, tracker = inicializar_modelos()

    cap = cv2.VideoCapture(config.VIDEO_IN)
    fps        = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    vid_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    panel_width  = 400
    canvas_width = vid_width + panel_width
    out = cv2.VideoWriter(config.VIDEO_OUT, cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (canvas_width, vid_height))

    # CSV detalle
    os.makedirs(os.path.dirname(config.CSV_OUT) or '.', exist_ok=True)
    f_csv = open(config.CSV_OUT, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(f_csv)
    csv_writer.writerow(['Frame', 'ID_Estable', 'ID_Tracker',
                         'X1', 'Y1', 'X2', 'Y2', 'CX', 'CY',
                         'En_Oclusion', 'Estado', 'Confirmado'])

    # ROI del YAML (o fallback)
    poly_escaner, poly_bolsa = cargar_roi_para_video(config.VIDEO_IN)

    recovery          = IdRecoverySystem()
    historial_prendas = {}
    estado_prendas    = {}
    ids_guardados     = set()
    frame_count       = 0

    print("Procesando con StrongSORT + ReID Híbrido v3 (ROI YAML)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        canvas, _, _ = crear_lienzo(frame, vid_width, vid_height, panel_width)
        alerta_global = False

        # ── 1. Detección RT-DETR ─────────────────────────────────────────────
        results = model.predict(
            frame,
            conf=config.DET_CONFIDENCE,
            iou=0.15,
            agnostic_nms=True,
            device=config.DEVICE,
            verbose=False,
            classes=config.TARGET_CLASSES,
            half=config.USE_FP16,
        )

        raw_det = []
        tracks  = []

        if len(results[0].boxes) > 0:
            cajas_crudas = results[0].boxes.data.cpu().numpy()

            boxes_nms = []; confs_nms = []
            for caja in cajas_crudas:
                rx1, ry1, rx2, ry2, rconf, _ = caja
                boxes_nms.append([int(rx1), int(ry1), int(rx2-rx1), int(ry2-ry1)])
                confs_nms.append(float(rconf))

            indices = cv2.dnn.NMSBoxes(boxes_nms, confs_nms,
                                       score_threshold=config.DET_CONFIDENCE,
                                       nms_threshold=config.NMS_IOU)

            raw_filtrado = []
            if len(indices) > 0:
                for i in indices.flatten():
                    caja = cajas_crudas[i]
                    rx1, ry1, rx2, ry2 = caja[:4]
                    if (rx2 - rx1) * (ry2 - ry1) >= config.MIN_AREA:
                        raw_filtrado.append(caja)

            raw_det = np.array(raw_filtrado)

            if len(raw_det) > 0:
                tracks_raw = tracker.update(raw_det, frame)

                tracks_estables = []
                for t in tracks_raw:
                    x1, y1, x2, y2, tracker_id, conf, cls, ind = t
                    tracker_id = int(tracker_id)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    id_est = recovery.get_id_estable(
                        tracker_id, cx, cy, frame_count,
                        frame, (x1, y1, x2, y2)
                    )
                    tracks_estables.append(
                        (x1, y1, x2, y2, id_est, tracker_id, conf, cls, cx, cy)
                    )

                ids_en_oclusion = set()
                for i in range(len(tracks_estables)):
                    for j in range(i + 1, len(tracks_estables)):
                        ta = tracks_estables[i]; tb = tracks_estables[j]
                        if calcular_iou(ta[:4], tb[:4]) >= config.OCCLUSION_IOU_THRESH:
                            ids_en_oclusion.add(ta[4])
                            ids_en_oclusion.add(tb[4])

                for (x1, y1, x2, y2, id_est, tracker_id, conf, cls, cx, cy) in tracks_estables:
                    en_oclusion = id_est in ids_en_oclusion
                    confirmado  = recovery.confirmado.get(id_est, False)

                    if id_est not in historial_prendas:
                        historial_prendas[id_est] = {
                            "paso_escaner":       False,
                            "estado":             "Detectando...",
                            "frames_sospechosos": 0,
                            "frames_visible":     0,
                            "ultima_pos":         (cx, cy),
                        }
                    else:
                        historial_prendas[id_est]["ultima_pos"] = (cx, cy)

                    historial_prendas[id_est]["frames_visible"] += 1
                    warmup_ok = (historial_prendas[id_est]["frames_visible"]
                                 >= config.ZONE_WARMUP_FRAMES)

                    if warmup_ok and not en_oclusion:
                        en_escaner = cv2.pointPolygonTest(poly_escaner, (cx, cy), False) >= 0
                        en_bolsa   = cv2.pointPolygonTest(poly_bolsa,   (cx, cy), False) >= 0

                        if en_escaner:
                            historial_prendas[id_est]["paso_escaner"]      = True
                            historial_prendas[id_est]["estado"]             = "Escaneando..."
                            historial_prendas[id_est]["frames_sospechosos"] = 0
                        elif en_bolsa:
                            if not historial_prendas[id_est]["paso_escaner"]:
                                historial_prendas[id_est]["frames_sospechosos"] += 1
                                if (historial_prendas[id_est]["frames_sospechosos"]
                                        >= config.ZONE_ALERT_FRAMES):
                                    historial_prendas[id_est]["estado"] = "ALERTA: EVASION"
                                    alerta_global = True
                                else:
                                    historial_prendas[id_est]["estado"] = "Evaluando..."
                            else:
                                historial_prendas[id_est]["estado"]             = "Desalarmado OK"
                                historial_prendas[id_est]["frames_sospechosos"] = 0
                        else:
                            if historial_prendas[id_est]["estado"] == "Evaluando...":
                                historial_prendas[id_est]["frames_sospechosos"] = 0
                                historial_prendas[id_est]["estado"] = "Detectando..."
                    elif en_oclusion:
                        historial_prendas[id_est]["estado"] = "Ocluida..."

                    estado_prendas[id_est] = historial_prendas[id_est]["estado"]

                    csv_writer.writerow([
                        frame_count, id_est, tracker_id,
                        int(x1), int(y1), int(x2), int(y2),
                        cx, cy,
                        int(en_oclusion),
                        estado_prendas[id_est],
                        int(confirmado),
                    ])

                    # Guardar primer recorte
                    if id_est not in ids_guardados:
                        ix1 = max(0, int(x1)); iy1 = max(0, int(y1))
                        ix2 = min(vid_width, int(x2)); iy2 = min(vid_height, int(y2))
                        crop = frame[iy1:iy2, ix1:ix2]
                        if crop.size > 0:
                            cv2.imwrite(
                                os.path.join(config.CROPS_DIR, f"ropa_id_{id_est}.jpg"),
                                crop
                            )
                            ids_guardados.add(id_est)

                tracks = np.array([
                    [x1, y1, x2, y2, id_est, conf, cls, 0]
                    for (x1, y1, x2, y2, id_est, tracker_id, conf, cls, cx, cy)
                    in tracks_estables
                ]) if tracks_estables else []

        ids_unicos = recovery.get_ids_estables_activos()
        canvas = dibujar_cajas(canvas, raw_det, tracks, estado_prendas,
                               poly_escaner=poly_escaner, poly_bolsa=poly_bolsa)
        canvas = dibujar_panel(canvas, vid_width, canvas_width, frame_count,
                               len(raw_det),
                               len(tracks) if hasattr(tracks, '__len__') else 0,
                               len(ids_unicos))
        out.write(canvas)

    cap.release()
    out.release()
    f_csv.close()
    print(f"Procesamiento finalizado v3! | IDs únicos: {len(ids_unicos)}")


if __name__ == "__main__":
    main()