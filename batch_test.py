"""
batch_test.py  —  v2
Corre el sistema de tracking (StrongSORT + ReID Híbrido) sobre video0..video6.
Genera archivos con prefijo 'v2_' para no sobreescribir los resultados anteriores.
"""

import cv2
import os
import numpy as np
import math
import csv
import time

import config
from modules.vision import inicializar_modelos
from modules.ui import crear_lienzo, dibujar_cajas, dibujar_panel


# ──────────────────────────────────────────────
VIDEOS = [f"videos_para_testear/video{i}.mp4" for i in range(7)]
PREFIX = "v2"
# ──────────────────────────────────────────────


def calcular_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter)


class IdRecoverySystem:
    def __init__(self):
        self.id_map       = {}
        self.id_inverso   = {}
        self.ultima_pos   = {}
        self.ultimo_frame = {}
        self.histograma   = {}
        self.siguiente_id = 1

    def _extraer_histograma(self, frame, x1, y1, x2, y2):
        ix1=max(0,int(x1)); iy1=max(0,int(y1))
        ix2=min(frame.shape[1],int(x2)); iy2=min(frame.shape[0],int(y2))
        crop = frame[iy1:iy2, ix1:ix2]
        if crop.size == 0: return None
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv],[0,1],None,[18,8],[0,180,0,256])
        cv2.normalize(hist, hist)
        return hist

    def _similitud(self, ha, hb):
        if ha is None or hb is None: return 0.5
        return float(max(0.0, cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)))

    def _actualizar_hist(self, id_est, h):
        if h is None: return
        a = config.TRACK_EMA_ALPHA
        if id_est in self.histograma and self.histograma[id_est] is not None:
            self.histograma[id_est] = a*self.histograma[id_est] + (1-a)*h
        else:
            self.histograma[id_est] = h.copy()

    def get_id_estable(self, tracker_id, cx, cy, frame_count, frame, bbox):
        if tracker_id in self.id_map:
            id_est = self.id_map[tracker_id]
            self.ultima_pos[id_est]   = (cx, cy)
            self.ultimo_frame[id_est] = frame_count
            self._actualizar_hist(id_est, self._extraer_histograma(frame, *bbox))
            return id_est

        hist_nuevo = self._extraer_histograma(frame, *bbox)
        mejor_id = None; mejor_score = -float('inf')

        for id_est, (ox, oy) in self.ultima_pos.items():
            if id_est in self.id_inverso and self.id_inverso[id_est] in self.id_map:
                continue
            if frame_count - self.ultimo_frame.get(id_est,0) > config.ID_RECOVERY_MAX_AGE:
                continue
            dist = math.hypot(cx-ox, cy-oy)
            if dist > config.ID_RECOVERY_MAX_DIST: continue
            s = (config.ID_RECOVERY_SPATIAL_WEIGHT    * (1 - dist/config.ID_RECOVERY_MAX_DIST)
               + config.ID_RECOVERY_APPEARANCE_WEIGHT * self._similitud(hist_nuevo, self.histograma.get(id_est)))
            if s > mejor_score:
                mejor_score = s; mejor_id = id_est

        id_est = mejor_id if (mejor_id and mejor_score >= config.ID_RECOVERY_SCORE_THRESHOLD) \
                          else self.siguiente_id
        if id_est == self.siguiente_id:
            self.siguiente_id += 1

        self.id_map[tracker_id]     = id_est
        self.id_inverso[id_est]     = tracker_id
        self.ultima_pos[id_est]     = (cx, cy)
        self.ultimo_frame[id_est]   = frame_count
        self._actualizar_hist(id_est, hist_nuevo)
        return id_est

    def get_ids_estables_activos(self):
        return set(self.id_map.values())


def procesar_video(video_in, video_out, csv_out, model, tracker):
    if not os.path.exists(video_in):
        print(f"  [SKIP] No encontrado: {video_in}")
        return

    cap = cv2.VideoCapture(video_in)
    fps        = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    vid_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    panel_width  = 400
    canvas_width = vid_width + panel_width
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (canvas_width, vid_height))

    f_csv = open(csv_out, mode='w', newline='', encoding='utf-8')
    w     = csv.writer(f_csv)
    w.writerow(['Frame','ID_Estable','ID_Tracker','X1','Y1','X2','Y2',
                'CX','CY','En_Oclusion','Estado'])

    recovery          = IdRecoverySystem()
    historial_prendas = {}
    estado_prendas    = {}
    frame_count       = 0
    tracker.reset()

    poly_escaner = np.array(config.ZONA_ESCANER, np.int32)
    poly_bolsa   = np.array(config.ZONA_BOLSA,   np.int32)

    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        if frame_count % 30 == 0 or frame_count == 1:
            pct = frame_count/total_frames*100 if total_frames else 0
            print(f"    Frame {frame_count}/{total_frames} ({pct:.0f}%)  {time.time()-t0:.0f}s", end='\r')

        canvas, _, _ = crear_lienzo(frame, vid_width, vid_height, panel_width)

        results = model.predict(
            frame, conf=config.DET_CONFIDENCE, iou=0.15,
            agnostic_nms=True, device=config.DEVICE,
            verbose=False, classes=config.TARGET_CLASSES
        )

        raw_det = []; tracks = []

        if len(results[0].boxes) > 0:
            cajas_crudas = results[0].boxes.data.cpu().numpy()
            boxes_nms=[]; confs_nms=[]
            for caja in cajas_crudas:
                rx1,ry1,rx2,ry2,rconf,_ = caja
                boxes_nms.append([int(rx1),int(ry1),int(rx2-rx1),int(ry2-ry1)])
                confs_nms.append(float(rconf))

            indices = cv2.dnn.NMSBoxes(boxes_nms, confs_nms,
                                       score_threshold=config.DET_CONFIDENCE,
                                       nms_threshold=config.NMS_IOU)
            raw_filtrado = []
            if len(indices) > 0:
                for i in indices.flatten():
                    caja = cajas_crudas[i]
                    rx1,ry1,rx2,ry2 = caja[:4]
                    if (rx2-rx1)*(ry2-ry1) >= config.MIN_AREA:
                        raw_filtrado.append(caja)

            raw_det = np.array(raw_filtrado)

            if len(raw_det) > 0:
                tracks_raw = tracker.update(raw_det, frame)
                tracks_est = []
                for t in tracks_raw:
                    x1,y1,x2,y2,tid,conf,cls,ind = t
                    tid=int(tid)
                    cx=int((x1+x2)/2); cy=int((y1+y2)/2)
                    ide = recovery.get_id_estable(tid,cx,cy,frame_count,frame,(x1,y1,x2,y2))
                    tracks_est.append((x1,y1,x2,y2,ide,tid,conf,cls,cx,cy))

                # Oclusiones
                ids_ocl = set()
                for i in range(len(tracks_est)):
                    for j in range(i+1, len(tracks_est)):
                        if calcular_iou(tracks_est[i][:4], tracks_est[j][:4]) >= config.OCCLUSION_IOU_THRESH:
                            ids_ocl.add(tracks_est[i][4]); ids_ocl.add(tracks_est[j][4])

                for (x1,y1,x2,y2,ide,tid,conf,cls,cx,cy) in tracks_est:
                    en_ocl = ide in ids_ocl
                    if ide not in historial_prendas:
                        historial_prendas[ide] = {
                            "paso_escaner":False,"estado":"Detectando...",
                            "frames_sospechosos":0,"frames_visible":0,
                            "ultima_pos":(cx,cy)
                        }
                    else:
                        historial_prendas[ide]["ultima_pos"] = (cx,cy)

                    historial_prendas[ide]["frames_visible"] += 1
                    warmup = historial_prendas[ide]["frames_visible"] >= config.ZONE_WARMUP_FRAMES

                    if warmup and not en_ocl:
                        en_esc = cv2.pointPolygonTest(poly_escaner,(cx,cy),False) >= 0
                        en_bol = cv2.pointPolygonTest(poly_bolsa,  (cx,cy),False) >= 0
                        if en_esc:
                            historial_prendas[ide]["paso_escaner"]=True
                            historial_prendas[ide]["estado"]="Escaneando..."
                            historial_prendas[ide]["frames_sospechosos"]=0
                        elif en_bol:
                            if not historial_prendas[ide]["paso_escaner"]:
                                historial_prendas[ide]["frames_sospechosos"] += 1
                                if historial_prendas[ide]["frames_sospechosos"] >= config.ZONE_ALERT_FRAMES:
                                    historial_prendas[ide]["estado"]="ALERTA: EVASION"
                                else:
                                    historial_prendas[ide]["estado"]="Evaluando..."
                            else:
                                historial_prendas[ide]["estado"]="Desalarmado OK"
                                historial_prendas[ide]["frames_sospechosos"]=0
                        else:
                            if historial_prendas[ide]["estado"]=="Evaluando...":
                                historial_prendas[ide]["frames_sospechosos"]=0
                                historial_prendas[ide]["estado"]="Detectando..."
                    elif en_ocl:
                        historial_prendas[ide]["estado"]="Ocluida..."

                    estado_prendas[ide] = historial_prendas[ide]["estado"]
                    w.writerow([frame_count,ide,tid,int(x1),int(y1),int(x2),int(y2),
                                cx,cy,int(en_ocl),estado_prendas[ide]])

                tracks = np.array([
                    [x1,y1,x2,y2,ide,conf,cls,0]
                    for (x1,y1,x2,y2,ide,tid,conf,cls,cx,cy) in tracks_est
                ]) if tracks_est else []

        ids_u = recovery.get_ids_estables_activos()
        canvas = dibujar_cajas(canvas, raw_det, tracks, estado_prendas)
        canvas = dibujar_panel(canvas, vid_width, canvas_width, frame_count,
                               len(raw_det), len(tracks) if hasattr(tracks,'__len__') else 0,
                               len(ids_u))
        out.write(canvas)

    cap.release(); out.release(); f_csv.close()
    print(f"\n    ✓ {frame_count} frames en {time.time()-t0:.1f}s  →  {video_out}")


if __name__ == "__main__":
    os.makedirs("videos_salida", exist_ok=True)
    print("Cargando modelos (una sola vez)...")
    model, tracker = inicializar_modelos()
    print("Modelos listos. Iniciando batch v2...\n")

    for vp in VIDEOS:
        nombre    = os.path.splitext(os.path.basename(vp))[0]
        video_out = f"videos_salida/{PREFIX}_resultado_{nombre}.mp4"
        csv_out   = f"videos_salida/{PREFIX}_resultado_{nombre}.csv"
        print(f"▶  {vp}  →  {video_out}")
        procesar_video(vp, video_out, csv_out, model, tracker)

    print(f"\n✅ Batch v2 completo. Revisa la carpeta videos_salida/")
