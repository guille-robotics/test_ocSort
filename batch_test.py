"""
batch_test.py
Corre el sistema de tracking sobre video0..video6 de forma secuencial.
Genera un .mp4 y un .csv separado por cada video en la carpeta videos_salida/.
"""

import cv2
import os
import numpy as np
import math
import csv
import time

import config  # Solo para leer DEVICE, DETECTOR_WEIGHTS, etc.
from modules.vision import inicializar_modelos
from modules.ui import crear_lienzo, dibujar_cajas, dibujar_panel


# ──────────────────────────────────────────────
# VIDEOS A PROCESAR
# ──────────────────────────────────────────────
VIDEOS = [f"videos_para_testear/video{i}.mp4" for i in range(7)]  # video0..video6


# ──────────────────────────────────────────────
# SISTEMA DE RECUPERACIÓN DE ID (igual que main.py)
# ──────────────────────────────────────────────
class IdRecoverySystem:
    def __init__(self):
        self.id_map = {}
        self.id_inverso = {}
        self.ultima_pos = {}
        self.ultimo_frame = {}
        self.siguiente_id = 1

    def get_id_estable(self, tracker_id, cx, cy, frame_count):
        if tracker_id in self.id_map:
            id_estable = self.id_map[tracker_id]
            self.ultima_pos[id_estable] = (cx, cy)
            self.ultimo_frame[id_estable] = frame_count
            return id_estable

        mejor_id = None
        mejor_dist = float('inf')
        for id_est, (ox, oy) in self.ultima_pos.items():
            if id_est in self.id_inverso and self.id_inverso[id_est] in self.id_map:
                continue
            frames_perdido = frame_count - self.ultimo_frame.get(id_est, 0)
            if frames_perdido > config.ID_RECOVERY_MAX_AGE:
                continue
            dist = math.hypot(cx - ox, cy - oy)
            if dist < mejor_dist:
                mejor_dist = dist
                mejor_id = id_est

        if mejor_id is not None and mejor_dist < config.ID_RECOVERY_MAX_DIST:
            id_estable = mejor_id
        else:
            id_estable = self.siguiente_id
            self.siguiente_id += 1

        self.id_map[tracker_id] = id_estable
        self.id_inverso[id_estable] = tracker_id
        self.ultima_pos[id_estable] = (cx, cy)
        self.ultimo_frame[id_estable] = frame_count
        return id_estable

    def get_ids_estables_activos(self):
        return set(self.id_map.values())


# ──────────────────────────────────────────────
# FUNCIÓN PRINCIPAL DE PROCESAMIENTO
# ──────────────────────────────────────────────
def procesar_video(video_in, video_out, csv_out, model, tracker):
    if not os.path.exists(video_in):
        print(f"  [SKIP] No encontrado: {video_in}")
        return

    cap = cv2.VideoCapture(video_in)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    vid_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    panel_width  = 400
    canvas_width = vid_width + panel_width
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (canvas_width, vid_height))

    f_csv = open(csv_out, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(f_csv)
    csv_writer.writerow(['Frame', 'ID_Estable', 'ID_Tracker',
                         'X1', 'Y1', 'X2', 'Y2', 'CX', 'CY', 'Estado'])

    recovery = IdRecoverySystem()
    historial_prendas = {}
    estado_prendas = {}
    frame_count = 0

    poly_escaner = np.array(config.ZONA_ESCANER, np.int32)
    poly_bolsa   = np.array(config.ZONA_BOLSA, np.int32)

    # Reiniciar el tracker para cada video
    tracker.reset()

    t_inicio = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Progreso en consola
        if frame_count % 30 == 0 or frame_count == 1:
            elapsed = time.time() - t_inicio
            pct = frame_count / total_frames * 100 if total_frames > 0 else 0
            print(f"    Frame {frame_count}/{total_frames} ({pct:.0f}%)  |  {elapsed:.1f}s transcurridos", end='\r')

        canvas, _, _ = crear_lienzo(frame, vid_width, vid_height, panel_width)
        alerta_global = False

        results = model.predict(
            frame,
            conf=config.DET_CONFIDENCE,
            iou=0.15,
            agnostic_nms=True,
            device=config.DEVICE,
            verbose=False,
            classes=config.TARGET_CLASSES
        )

        raw_det = []
        tracks  = []

        if len(results[0].boxes) > 0:
            cajas_crudas = results[0].boxes.data.cpu().numpy()

            boxes_nms = []
            confs_nms = []
            for caja in cajas_crudas:
                rx1, ry1, rx2, ry2, rconf, rcls = caja
                boxes_nms.append([int(rx1), int(ry1), int(rx2 - rx1), int(ry2 - ry1)])
                confs_nms.append(float(rconf))

            indices = cv2.dnn.NMSBoxes(boxes_nms, confs_nms,
                                       score_threshold=config.DET_CONFIDENCE,
                                       nms_threshold=0.3)

            raw_filtrado = []
            if len(indices) > 0:
                for i in indices.flatten():
                    caja = cajas_crudas[i]
                    rx1, ry1, rx2, ry2 = caja[:4]
                    if (rx2 - rx1) * (ry2 - ry1) > 25000:
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
                    id_estable = recovery.get_id_estable(tracker_id, cx, cy, frame_count)
                    tracks_estables.append((x1, y1, x2, y2, id_estable, tracker_id, conf, cls, cx, cy))

                for (x1, y1, x2, y2, id_estable, tracker_id, conf, cls, cx, cy) in tracks_estables:
                    if id_estable not in historial_prendas:
                        historial_prendas[id_estable] = {
                            "paso_escaner": False,
                            "estado": "Detectando...",
                            "frames_sospechosos": 0,
                            "ultima_pos": (cx, cy)
                        }
                    else:
                        historial_prendas[id_estable]["ultima_pos"] = (cx, cy)

                    en_escaner = cv2.pointPolygonTest(poly_escaner, (cx, cy), False) >= 0
                    en_bolsa   = cv2.pointPolygonTest(poly_bolsa,   (cx, cy), False) >= 0

                    if en_escaner:
                        historial_prendas[id_estable]["paso_escaner"] = True
                        historial_prendas[id_estable]["estado"] = "Escaneando..."
                        historial_prendas[id_estable]["frames_sospechosos"] = 0
                    elif en_bolsa:
                        if not historial_prendas[id_estable]["paso_escaner"]:
                            historial_prendas[id_estable]["frames_sospechosos"] += 1
                            if historial_prendas[id_estable]["frames_sospechosos"] > 15:
                                historial_prendas[id_estable]["estado"] = "ALERTA: EVASION"
                                alerta_global = True
                            else:
                                historial_prendas[id_estable]["estado"] = "Evaluando..."
                        else:
                            historial_prendas[id_estable]["estado"] = "Desalarmado OK"
                            historial_prendas[id_estable]["frames_sospechosos"] = 0

                    estado_prendas[id_estable] = historial_prendas[id_estable]["estado"]

                    csv_writer.writerow([frame_count, id_estable, tracker_id,
                                         int(x1), int(y1), int(x2), int(y2),
                                         cx, cy, estado_prendas[id_estable]])

                tracks = np.array([
                    [x1, y1, x2, y2, id_estable, conf, cls, 0]
                    for (x1, y1, x2, y2, id_estable, tracker_id, conf, cls, cx, cy) in tracks_estables
                ]) if tracks_estables else []

        ids_unicos = recovery.get_ids_estables_activos()
        canvas = dibujar_cajas(canvas, raw_det, tracks, estado_prendas)
        canvas = dibujar_panel(canvas, vid_width, canvas_width, frame_count,
                               len(raw_det), len(tracks) if len(tracks) > 0 else 0,
                               len(ids_unicos))
        out.write(canvas)

    cap.release()
    out.release()
    f_csv.close()

    elapsed_total = time.time() - t_inicio
    print(f"\n    Listo: {frame_count} frames en {elapsed_total:.1f}s  →  {video_out}")


# ──────────────────────────────────────────────
# PUNTO DE ENTRADA
# ──────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("videos_salida", exist_ok=True)

    print("Cargando modelos una sola vez...")
    model, tracker = inicializar_modelos()
    print("Modelos listos. Iniciando batch...\n")

    for video_path in VIDEOS:
        nombre = os.path.splitext(os.path.basename(video_path))[0]  # ej: "video0"
        video_out = f"videos_salida/resultado_{nombre}.mp4"
        csv_out   = f"videos_salida/resultado_{nombre}.csv"

        print(f"▶ Procesando {video_path}  →  {video_out}")
        procesar_video(video_path, video_out, csv_out, model, tracker)

    print("\n✅ Batch completo. Todos los videos procesados.")
