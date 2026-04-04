"""
batch_test.py  —  v4
=====================
Mejoras respecto a v3:
  • Tracker configurable: BoT-SORT (default) / ByteTrack / StrongSORT
  • IdConsolidador: fusiona IDs fragmentados → cuenta real de prendas
  • Frame resumen al final de cada video con la cuenta de prendas
  • Panel limpio: solo IDs confirmados, sin ruido de cajas rojas
  • Gráfico de barras de prendas por video (la métrica clave)
  • CSV resumen incluye Prendas_Reales
"""

import cv2
import os
import glob
import numpy as np
import math
import csv
import time
import yaml
from pathlib import Path

import config
from modules.vision import inicializar_modelos
from modules.ui import (crear_lienzo, dibujar_cajas, dibujar_panel,
                        crear_frame_resumen)
from modules.report import VideoStats, guardar_csv_resumen, generar_imagen_comparativa
from modules.consolidador import IdConsolidador


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL
# ──────────────────────────────────────────────────────────────────────────────

VIDEOS_DIR    = "videos_para_testear"
OUTPUT_DIR    = "videos_salida"
YAML_ROI      = "config.yaml"
PREFIX        = "v4"
RESUMEN_FRAMES = 90   # frames finales de resumen (~3 seg a 30fps)

CSV_RESUMEN   = os.path.join(OUTPUT_DIR, f"{PREFIX}_resumen_batch.csv")
IMG_REPORTE   = os.path.join(OUTPUT_DIR, f"{PREFIX}_reporte_comparativo.png")


# ──────────────────────────────────────────────────────────────────────────────
# ROI desde YAML
# ──────────────────────────────────────────────────────────────────────────────

def cargar_rois_yaml(yaml_path: str) -> dict:
    if not os.path.exists(yaml_path):
        print(f"[ROI] YAML no encontrado: {yaml_path}  →  fallback a config.py")
        return {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def obtener_roi_video(nombre_video: str, rois_yaml: dict):
    clave = Path(nombre_video).stem
    if clave in rois_yaml:
        e = np.array(rois_yaml[clave]["Zona_Escaner"], np.int32)
        b = np.array(rois_yaml[clave]["Zona_Bolsa"],   np.int32)
        print(f"  [ROI] '{clave}' → YAML")
    else:
        e = np.array(config.ZONA_ESCANER, np.int32)
        b = np.array(config.ZONA_BOLSA,   np.int32)
        print(f"  [ROI] '{clave}' → config.py (fallback)")
    return e, b


# ──────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────────────────────────────────────

def calcular_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    aA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (aA + aB - inter)


# ──────────────────────────────────────────────────────────────────────────────
# SISTEMA DE RECUPERACIÓN DE ID (complemento al tracker)
# ──────────────────────────────────────────────────────────────────────────────

class IdRecoverySystem:
    """
    Capa sobre el tracker base: mantiene IDs estables y filtra tracks de ruido.
    Compatible con BoT-SORT, StrongSORT y ByteTrack.
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

    def _extraer_hist(self, frame, x1, y1, x2, y2):
        ix1=max(0,int(x1)); iy1=max(0,int(y1))
        ix2=min(frame.shape[1],int(x2)); iy2=min(frame.shape[0],int(y2))
        crop = frame[iy1:iy2, ix1:ix2]
        if crop.size == 0: return None
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv],[0,1],None,[18,8],[0,180,0,256])
        cv2.normalize(hist, hist)
        return hist

    def _sim(self, ha, hb):
        if ha is None or hb is None: return 0.5
        return float(max(0.0, cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)))

    def _upd_hist(self, id_est, h):
        if h is None: return
        a = config.TRACK_EMA_ALPHA
        if id_est in self.histograma and self.histograma[id_est] is not None:
            self.histograma[id_est] = a*self.histograma[id_est] + (1-a)*h
        else:
            self.histograma[id_est] = h.copy()

    def get_id_estable(self, tracker_id, cx, cy, frame_count, frame, bbox):
        if tracker_id in self.id_map:
            ide = self.id_map[tracker_id]
            self.ultima_pos[ide]    = (cx, cy)
            self.ultimo_frame[ide]  = frame_count
            self.frames_vistos[ide] = self.frames_vistos.get(ide,0) + 1
            if self.frames_vistos[ide] >= config.TRACK_MIN_STABLE_FRAMES:
                self.confirmado[ide] = True
            self._upd_hist(ide, self._extraer_hist(frame,*bbox))
            return ide

        hn = self._extraer_hist(frame,*bbox)
        mejor_id=None; mejor_s=-float('inf')

        for ide,(ox,oy) in self.ultima_pos.items():
            if ide in self.id_inverso and self.id_inverso[ide] in self.id_map:
                continue
            if frame_count - self.ultimo_frame.get(ide,0) > config.ID_RECOVERY_MAX_AGE:
                continue
            dist = math.hypot(cx-ox,cy-oy)
            if dist > config.ID_RECOVERY_MAX_DIST: continue
            s = (config.ID_RECOVERY_SPATIAL_WEIGHT    * (1-dist/config.ID_RECOVERY_MAX_DIST)
               + config.ID_RECOVERY_APPEARANCE_WEIGHT * self._sim(hn, self.histograma.get(ide)))
            if s > mejor_s: mejor_s=s; mejor_id=ide

        if mejor_id and mejor_s >= config.ID_RECOVERY_SCORE_THRESHOLD:
            ide = mejor_id
        else:
            ide = self.siguiente_id; self.siguiente_id += 1

        self.id_map[tracker_id]    = ide
        self.id_inverso[ide]       = tracker_id
        self.ultima_pos[ide]       = (cx,cy)
        self.ultimo_frame[ide]     = frame_count
        self.frames_vistos[ide]    = self.frames_vistos.get(ide,0) + 1
        if self.frames_vistos[ide] >= config.TRACK_MIN_STABLE_FRAMES:
            self.confirmado[ide] = True
        self._upd_hist(ide, hn)
        return ide

    def get_ids_activos(self):
        return set(self.id_map.values())

    def get_ids_confirmados(self):
        return {i for i in self.id_map.values() if self.confirmado.get(i, False)}


# ──────────────────────────────────────────────────────────────────────────────
# PROCESAMIENTO DE UN VIDEO
# ──────────────────────────────────────────────────────────────────────────────

def procesar_video(video_in, video_out, csv_out,
                   model, tracker, poly_escaner, poly_bolsa) -> VideoStats:

    if not os.path.exists(video_in):
        print(f"  [SKIP] No encontrado: {video_in}")
        return None

    cap          = cv2.VideoCapture(video_in)
    fps          = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    vid_width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    panel_width  = 400
    canvas_width = vid_width + panel_width
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (canvas_width, vid_height))

    f_csv = open(csv_out, mode='w', newline='', encoding='utf-8')
    w_csv = csv.writer(f_csv)
    w_csv.writerow(['Frame','ID_Estable','ID_Tracker',
                    'X1','Y1','X2','Y2','CX','CY',
                    'En_Oclusion','Estado','Confirmado'])

    recovery          = IdRecoverySystem()
    consolidador      = IdConsolidador()
    historial_prendas = {}
    estado_prendas    = {}
    frame_count       = 0
    tracker.reset()

    nombre_video = Path(video_in).stem
    stats = VideoStats(nombre_video, fps, total_frames)
    stats.mark_start()
    t0 = time.time()

    # ── Prendas estimadas en tiempo real (progresivo) ──────────────────────────
    prendas_estimadas_rt = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        if frame_count % 30 == 0 or frame_count == 1:
            pct = frame_count/total_frames*100 if total_frames else 0
            elapsed = time.time()-t0
            pfps = frame_count/elapsed if elapsed>0 else 0
            print(f"    Frame {frame_count}/{total_frames} ({pct:.0f}%)  "
                  f"{elapsed:.0f}s  [{pfps:.1f}fps]  "
                  f"prendas≈{prendas_estimadas_rt}", end='\r')

        canvas, _, _ = crear_lienzo(frame, vid_width, vid_height, panel_width)
        alerta_global = False

        # ── 1. RT-DETR ────────────────────────────────────────────────────────
        results = model.predict(
            frame, conf=config.DET_CONFIDENCE, iou=0.15,
            agnostic_nms=True, device=config.DEVICE,
            verbose=False, classes=config.TARGET_CLASSES,
            half=config.USE_FP16,
        )

        raw_det = []; tracks = []

        if len(results[0].boxes) > 0:
            cajas = results[0].boxes.data.cpu().numpy()

            # NMS adicional
            boxes_nms=[]; confs_nms=[]
            for c in cajas:
                rx1,ry1,rx2,ry2,rconf,_ = c
                boxes_nms.append([int(rx1),int(ry1),int(rx2-rx1),int(ry2-ry1)])
                confs_nms.append(float(rconf))
            idx = cv2.dnn.NMSBoxes(boxes_nms, confs_nms,
                                   score_threshold=config.DET_CONFIDENCE,
                                   nms_threshold=config.NMS_IOU)

            raw_f = []
            if len(idx) > 0:
                for i in idx.flatten():
                    c = cajas[i]; rx1,ry1,rx2,ry2 = c[:4]
                    if (rx2-rx1)*(ry2-ry1) >= config.MIN_AREA:
                        raw_f.append(c)
            raw_det = np.array(raw_f)

            if len(raw_det) > 0:
                tracks_raw = tracker.update(raw_det, frame)

                # IDs estables
                tracks_est = []
                for t in tracks_raw:
                    x1,y1,x2,y2,tid,conf,cls,ind = t
                    tid=int(tid); cx=int((x1+x2)/2); cy=int((y1+y2)/2)
                    ide = recovery.get_id_estable(tid,cx,cy,frame_count,frame,(x1,y1,x2,y2))
                    tracks_est.append((x1,y1,x2,y2,ide,tid,conf,cls,cx,cy))

                # Oclusiones
                ids_ocl = set()
                for i in range(len(tracks_est)):
                    for j in range(i+1,len(tracks_est)):
                        if calcular_iou(tracks_est[i][:4],tracks_est[j][:4]) >= config.OCCLUSION_IOU_THRESH:
                            ids_ocl.add(tracks_est[i][4]); ids_ocl.add(tracks_est[j][4])

                # Lógica de zonas + registrar en consolidador
                for (x1,y1,x2,y2,ide,tid,conf,cls,cx,cy) in tracks_est:
                    en_ocl     = ide in ids_ocl
                    confirmado = recovery.confirmado.get(ide, False)
                    hist_ide   = recovery.histograma.get(ide)

                    # Registrar en consolidador
                    consolidador.registrar(ide, frame_count, hist_ide, cx, cy, confirmado)

                    if ide not in historial_prendas:
                        historial_prendas[ide] = {
                            "paso_escaner":False,"estado":"Detectando...",
                            "frames_sospechosos":0,"frames_visible":0,"ultima_pos":(cx,cy)
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
                                    alerta_global=True
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
                    w_csv.writerow([frame_count,ide,tid,
                                    int(x1),int(y1),int(x2),int(y2),
                                    cx,cy,int(en_ocl),estado_prendas[ide],int(confirmado)])

                tracks = np.array([
                    [x1,y1,x2,y2,ide,conf,cls,0]
                    for (x1,y1,x2,y2,ide,tid,conf,cls,cx,cy) in tracks_est
                ]) if tracks_est else []

        ids_todos      = recovery.get_ids_activos()
        ids_conf       = recovery.get_ids_confirmados()

        # Estimación en tiempo real de prendas (consolidación parcial cada 30f)
        if frame_count % 30 == 0:
            _, prendas_estimadas_rt, _ = consolidador.consolidar()

        stats.update(
            raw_count=len(raw_det),
            track_count=len(tracks) if hasattr(tracks,'__len__') else 0,
            ids_activos=ids_todos,
            alerta_global=alerta_global,
            estado_prendas=estado_prendas,
        )

        canvas = dibujar_cajas(canvas, raw_det, tracks, estado_prendas,
                               ids_confirmados=ids_conf,
                               poly_escaner=poly_escaner, poly_bolsa=poly_bolsa)
        canvas = dibujar_panel(canvas, vid_width, canvas_width, frame_count,
                               len(raw_det),
                               len(tracks) if hasattr(tracks,'__len__') else 0,
                               len(ids_todos),
                               prendas_reales=prendas_estimadas_rt)
        out.write(canvas)

    cap.release()

    # ── Consolidación FINAL ────────────────────────────────────────────────────
    grupos, prendas_reales, ids_validos = consolidador.consolidar()
    resumen_p = consolidador.resumen_prendas(grupos)
    stats.prendas_reales = prendas_reales

    # ── Frames de resumen al final del video ──────────────────────────────────
    stats.mark_end()
    frame_resumen = crear_frame_resumen(
        vid_width, vid_height, panel_width,
        nombre_video,
        prendas_reales,
        resumen_p,
        len(ids_todos),
        proc_fps=stats.proc_fps,
        duracion_seg=stats.duracion_seg,
    )
    for _ in range(RESUMEN_FRAMES):
        out.write(frame_resumen)

    out.release()
    f_csv.close()

    elapsed = time.time()-t0
    print(f"\n    ✓ {frame_count} frames | {elapsed:.1f}s | "
          f"{stats.proc_fps:.1f}fps | "
          f"IDs raw={len(ids_todos)} | "
          f"PRENDAS={prendas_reales} → {video_out}")

    # Imprimir detalle de prendas
    if resumen_p:
        print(f"    Detalle de prendas consolidadas:")
        for p in resumen_p:
            print(f"      Prenda {p['prenda']}: frames {p['first_frame']}→{p['last_frame']} "
                  f"| IDs fusionados: {p['ids']} | frames visibles: {p['frames_visible']}")

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    patron = os.path.join(VIDEOS_DIR, "video*.mp4")
    videos = sorted(glob.glob(patron),
                    key=lambda p: int(''.join(filter(str.isdigit, Path(p).stem)) or '0'))

    if not videos:
        print(f"[ERROR] No se encontraron videos en '{VIDEOS_DIR}/'"); exit(1)

    print(f"[batch] {len(videos)} videos | Tracker: {config.TRACKER_TYPE.upper()}\n")

    rois_yaml = cargar_rois_yaml(YAML_ROI)

    print("Cargando modelos...")
    model, tracker = inicializar_modelos()
    print("Modelos listos. Iniciando batch v4...\n")

    all_stats = []
    for vp in videos:
        nombre    = Path(vp).stem
        video_out = os.path.join(OUTPUT_DIR, f"{PREFIX}_resultado_{nombre}.mp4")
        csv_out   = os.path.join(OUTPUT_DIR, f"{PREFIX}_resultado_{nombre}.csv")
        poly_e, poly_b = obtener_roi_video(vp, rois_yaml)

        print(f"\n▶  {vp}  →  {video_out}")
        s = procesar_video(vp, video_out, csv_out, model, tracker, poly_e, poly_b)
        if s: all_stats.append(s)

    if all_stats:
        guardar_csv_resumen(all_stats, CSV_RESUMEN)
        generar_imagen_comparativa(all_stats, IMG_REPORTE)

    print(f"\n✅ Batch v4 completo.")
    print(f"   Tracker usado    : {config.TRACKER_TYPE.upper()}")
    print(f"   Videos procesados: {len(all_stats)}")
    print(f"   CSV resumen      : {CSV_RESUMEN}")
    print(f"   Imagen reporte   : {IMG_REPORTE}")
    print()
    print(f"{'Video':<12} {'Prendas':>8} {'IDs raw':>8} {'FPS':>6} {'%Alert':>8}")
    print("─" * 50)
    for s in all_stats:
        print(f"{s.video_name:<12} {s.prendas_reales:>8} "
              f"{s.total_ids_unicos:>8} {s.proc_fps:>6.1f} "
              f"{s.alertas_pct:>7.1f}%")
