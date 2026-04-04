"""
batch_test.py  —  v3
=====================
Mejoras respecto a v2:
  • Auto-detecta TODOS los videos en videos_para_testear/ (sin límite fijo).
  • Carga ROI (Zona_Bolsa / Zona_Escaner) individuales por video desde config.yaml.
  • Fallback a config.py si el video no tiene ROI definida en el YAML.
  • Genera CSV de detalle + CSV de resumen + imagen PNG comparativa.
  • Parámetros de tracking más estrictos para reducir IDs espurios.
  • Media-pipe de detección con fp16 habilitado cuando hay GPU.
  • Reset correcto del tracker entre videos.
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
from modules.ui import crear_lienzo, dibujar_cajas, dibujar_panel
from modules.report import VideoStats, guardar_csv_resumen, generar_imagen_comparativa


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GLOBAL DEL BATCH
# ──────────────────────────────────────────────────────────────────────────────

VIDEOS_DIR   = "videos_para_testear"
OUTPUT_DIR   = "videos_salida"
YAML_ROI     = "config.yaml"           # ROIs individuales por video
PREFIX       = "v3"                    # prefijo de archivos de salida

# Salidas de resumen
CSV_RESUMEN  = os.path.join(OUTPUT_DIR, f"{PREFIX}_resumen_batch.csv")
IMG_REPORTE  = os.path.join(OUTPUT_DIR, f"{PREFIX}_reporte_comparativo.png")


# ──────────────────────────────────────────────────────────────────────────────
# CARGA DE ROIs DESDE YAML
# ──────────────────────────────────────────────────────────────────────────────

def cargar_rois_yaml(yaml_path: str) -> dict:
    """
    Devuelve un dict  { "video0": {"Zona_Bolsa": [...], "Zona_Escaner": [...]}, ... }
    Si el archivo no existe devuelve {}.
    """
    if not os.path.exists(yaml_path):
        print(f"[ROI] YAML no encontrado: {yaml_path}  →  se usarán ROIs de config.py")
        return {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    print(f"[ROI] YAML cargado: {yaml_path}  ({len(data)} entradas)")
    return data


def obtener_roi_video(nombre_video: str, rois_yaml: dict):
    """
    Devuelve (poly_escaner, poly_bolsa) como np.array int32.
    Busca en el YAML con la clave 'videoN' (extraída del nombre de archivo).
    Si no encuentra, usa los fallback de config.py.
    """
    # Extraer clave: "video3.mp4" → "video3"
    clave = Path(nombre_video).stem   # e.g. "video3"

    if clave in rois_yaml:
        datos = rois_yaml[clave]
        escaner = np.array(datos["Zona_Escaner"], np.int32)
        bolsa   = np.array(datos["Zona_Bolsa"],   np.int32)
        print(f"  [ROI] '{clave}' → ROI personalizado del YAML")
    else:
        escaner = np.array(config.ZONA_ESCANER, np.int32)
        bolsa   = np.array(config.ZONA_BOLSA,   np.int32)
        print(f"  [ROI] '{clave}' → ROI por defecto de config.py")

    return escaner, bolsa


# ──────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────────────────────────────────────

def calcular_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter)


# ──────────────────────────────────────────────────────────────────────────────
# SISTEMA DE RECUPERACIÓN DE ID HÍBRIDO
# ──────────────────────────────────────────────────────────────────────────────

class IdRecoverySystem:
    """
    Mantiene IDs estables a lo largo del video, incluso cuando StrongSORT
    reasigna el tracker_id tras una oclusión.
    Estrategia combinada: distancia espacial + similitud de histograma HSV.

    v3: agrega supresión de IDs efímeros — un ID solo se "confirma" tras
    TRACK_MIN_STABLE_FRAMES frames continuos, evitando contar tracks ruido
    de 1-2 frames como IDs reales.
    """

    def __init__(self):
        self.id_map          = {}   # tracker_id  → id_estable
        self.id_inverso      = {}   # id_estable  → tracker_id activo
        self.ultima_pos      = {}   # id_estable  → (cx, cy)
        self.ultimo_frame    = {}   # id_estable  → último frame visible
        self.histograma      = {}   # id_estable  → histograma HSV (EMA)
        self.frames_vistos   = {}   # id_estable  → frames consecutivos vistos
        self.confirmado      = {}   # id_estable  → bool (superó warmup)
        self.siguiente_id    = 1

    # ── extracción de histograma ──────────────────────────────────────────────
    def _extraer_histograma(self, frame, x1, y1, x2, y2):
        ix1=max(0,int(x1)); iy1=max(0,int(y1))
        ix2=min(frame.shape[1],int(x2)); iy2=min(frame.shape[0],int(y2))
        crop = frame[iy1:iy2, ix1:ix2]
        if crop.size == 0:
            return None
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0,1], None, [18,8], [0,180,0,256])
        cv2.normalize(hist, hist)
        return hist

    def _similitud(self, ha, hb):
        if ha is None or hb is None:
            return 0.5
        return float(max(0.0, cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)))

    def _actualizar_hist(self, id_est, h):
        if h is None:
            return
        a = config.TRACK_EMA_ALPHA
        if id_est in self.histograma and self.histograma[id_est] is not None:
            self.histograma[id_est] = a * self.histograma[id_est] + (1-a) * h
        else:
            self.histograma[id_est] = h.copy()

    # ── asignación de ID estable ──────────────────────────────────────────────
    def get_id_estable(self, tracker_id, cx, cy, frame_count, frame, bbox):
        # Caso 1: tracker_id ya conocido → solo actualizar
        if tracker_id in self.id_map:
            id_est = self.id_map[tracker_id]
            self.ultima_pos[id_est]   = (cx, cy)
            self.ultimo_frame[id_est] = frame_count
            self.frames_vistos[id_est] = self.frames_vistos.get(id_est, 0) + 1
            if self.frames_vistos[id_est] >= config.TRACK_MIN_STABLE_FRAMES:
                self.confirmado[id_est] = True
            self._actualizar_hist(id_est, self._extraer_histograma(frame, *bbox))
            return id_est

        # Caso 2: tracker_id NUEVO → buscar candidato perdido
        hist_nuevo = self._extraer_histograma(frame, *bbox)
        mejor_id = None; mejor_score = -float('inf')

        for id_est, (ox, oy) in self.ultima_pos.items():
            # ¿Ya tiene tracker activo?
            if id_est in self.id_inverso and self.id_inverso[id_est] in self.id_map:
                continue
            # ¿Lleva demasiado tiempo perdido?
            if frame_count - self.ultimo_frame.get(id_est, 0) > config.ID_RECOVERY_MAX_AGE:
                continue
            dist = math.hypot(cx-ox, cy-oy)
            if dist > config.ID_RECOVERY_MAX_DIST:
                continue

            s = (config.ID_RECOVERY_SPATIAL_WEIGHT    * (1 - dist/config.ID_RECOVERY_MAX_DIST)
               + config.ID_RECOVERY_APPEARANCE_WEIGHT * self._similitud(hist_nuevo,
                                                                         self.histograma.get(id_est)))
            if s > mejor_score:
                mejor_score = s; mejor_id = id_est

        if mejor_id is not None and mejor_score >= config.ID_RECOVERY_SCORE_THRESHOLD:
            id_est = mejor_id
        else:
            id_est = self.siguiente_id
            self.siguiente_id += 1

        self.id_map[tracker_id]    = id_est
        self.id_inverso[id_est]    = tracker_id
        self.ultima_pos[id_est]    = (cx, cy)
        self.ultimo_frame[id_est]  = frame_count
        self.frames_vistos[id_est] = self.frames_vistos.get(id_est, 0) + 1
        if self.frames_vistos[id_est] >= config.TRACK_MIN_STABLE_FRAMES:
            self.confirmado[id_est] = True
        self._actualizar_hist(id_est, hist_nuevo)
        return id_est

    def id_esta_confirmado(self, id_est):
        return self.confirmado.get(id_est, False)

    def get_ids_estables_activos(self):
        return set(self.id_map.values())

    def get_ids_confirmados_activos(self):
        """Solo IDs que superaron el warmup mínimo de frames."""
        return {i for i in self.id_map.values() if self.confirmado.get(i, False)}


# ──────────────────────────────────────────────────────────────────────────────
# PROCESAMIENTO DE UN VIDEO
# ──────────────────────────────────────────────────────────────────────────────

def procesar_video(video_in, video_out, csv_out,
                   model, tracker, poly_escaner, poly_bolsa) -> VideoStats:
    """Procesa un video completo y devuelve las estadísticas acumuladas."""

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

    # ── CSV de detalle (fila por detección por frame) ─────────────────────────
    f_csv = open(csv_out, mode='w', newline='', encoding='utf-8')
    w_csv = csv.writer(f_csv)
    w_csv.writerow([
        'Frame', 'ID_Estable', 'ID_Tracker', 'X1', 'Y1', 'X2', 'Y2',
        'CX', 'CY', 'En_Oclusion', 'Estado', 'Confirmado'
    ])

    # ── Estado inicial ────────────────────────────────────────────────────────
    recovery          = IdRecoverySystem()
    historial_prendas = {}
    estado_prendas    = {}
    frame_count       = 0
    tracker.reset()

    nombre_video = Path(video_in).stem
    stats = VideoStats(nombre_video, fps, total_frames)
    stats.mark_start()

    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # ── Progreso en consola ───────────────────────────────────────────────
        if frame_count % 30 == 0 or frame_count == 1:
            pct = frame_count / total_frames * 100 if total_frames else 0
            elapsed = time.time() - t0
            proc_fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"    Frame {frame_count}/{total_frames} ({pct:.0f}%)  "
                  f"{elapsed:.0f}s  [{proc_fps:.1f} fps proc]", end='\r')

        canvas, _, _ = crear_lienzo(frame, vid_width, vid_height, panel_width)
        alerta_global = False

        # ── 1. Detección RT-DETR ──────────────────────────────────────────────
        results = model.predict(
            frame,
            conf=config.DET_CONFIDENCE,
            iou=0.15,
            agnostic_nms=True,
            device=config.DEVICE,
            verbose=False,
            classes=config.TARGET_CLASSES,
            half=config.USE_FP16,            # fp16 en GPU para mayor velocidad
        )

        raw_det = []
        tracks  = []

        if len(results[0].boxes) > 0:
            cajas_crudas = results[0].boxes.data.cpu().numpy()

            # ── 2. NMS manual adicional ───────────────────────────────────────
            boxes_nms = []; confs_nms = []
            for caja in cajas_crudas:
                rx1, ry1, rx2, ry2, rconf, _ = caja
                boxes_nms.append([int(rx1), int(ry1), int(rx2-rx1), int(ry2-ry1)])
                confs_nms.append(float(rconf))

            indices = cv2.dnn.NMSBoxes(
                boxes_nms, confs_nms,
                score_threshold=config.DET_CONFIDENCE,
                nms_threshold=config.NMS_IOU
            )

            # ── 3. Filtro de área ─────────────────────────────────────────────
            raw_filtrado = []
            if len(indices) > 0:
                for i in indices.flatten():
                    caja = cajas_crudas[i]
                    rx1, ry1, rx2, ry2 = caja[:4]
                    if (rx2-rx1) * (ry2-ry1) >= config.MIN_AREA:
                        raw_filtrado.append(caja)

            raw_det = np.array(raw_filtrado)

            if len(raw_det) > 0:
                tracks_raw = tracker.update(raw_det, frame)

                # ── 4. IDs estables ───────────────────────────────────────────
                tracks_est = []
                for t in tracks_raw:
                    x1, y1, x2, y2, tid, conf, cls, ind = t
                    tid = int(tid)
                    cx  = int((x1+x2)/2)
                    cy  = int((y1+y2)/2)
                    ide = recovery.get_id_estable(
                        tid, cx, cy, frame_count, frame, (x1,y1,x2,y2)
                    )
                    tracks_est.append((x1,y1,x2,y2, ide, tid, conf, cls, cx, cy))

                # ── 5. Oclusiones ─────────────────────────────────────────────
                ids_ocl = set()
                for i in range(len(tracks_est)):
                    for j in range(i+1, len(tracks_est)):
                        if calcular_iou(tracks_est[i][:4], tracks_est[j][:4]) >= config.OCCLUSION_IOU_THRESH:
                            ids_ocl.add(tracks_est[i][4])
                            ids_ocl.add(tracks_est[j][4])

                # ── 6. Lógica de zonas y estado ───────────────────────────────
                for (x1,y1,x2,y2, ide, tid, conf, cls, cx, cy) in tracks_est:

                    en_ocl      = ide in ids_ocl
                    confirmado  = recovery.id_esta_confirmado(ide)

                    if ide not in historial_prendas:
                        historial_prendas[ide] = {
                            "paso_escaner":        False,
                            "estado":              "Detectando...",
                            "frames_sospechosos":  0,
                            "frames_visible":      0,
                            "ultima_pos":          (cx, cy),
                        }
                    else:
                        historial_prendas[ide]["ultima_pos"] = (cx, cy)

                    historial_prendas[ide]["frames_visible"] += 1
                    warmup = historial_prendas[ide]["frames_visible"] >= config.ZONE_WARMUP_FRAMES

                    if warmup and not en_ocl:
                        en_esc = cv2.pointPolygonTest(poly_escaner, (cx,cy), False) >= 0
                        en_bol = cv2.pointPolygonTest(poly_bolsa,   (cx,cy), False) >= 0

                        if en_esc:
                            historial_prendas[ide]["paso_escaner"]      = True
                            historial_prendas[ide]["estado"]             = "Escaneando..."
                            historial_prendas[ide]["frames_sospechosos"] = 0
                        elif en_bol:
                            if not historial_prendas[ide]["paso_escaner"]:
                                historial_prendas[ide]["frames_sospechosos"] += 1
                                if historial_prendas[ide]["frames_sospechosos"] >= config.ZONE_ALERT_FRAMES:
                                    historial_prendas[ide]["estado"] = "ALERTA: EVASION"
                                    alerta_global = True
                                else:
                                    historial_prendas[ide]["estado"] = "Evaluando..."
                            else:
                                historial_prendas[ide]["estado"]             = "Desalarmado OK"
                                historial_prendas[ide]["frames_sospechosos"] = 0
                        else:
                            if historial_prendas[ide]["estado"] == "Evaluando...":
                                historial_prendas[ide]["frames_sospechosos"] = 0
                                historial_prendas[ide]["estado"] = "Detectando..."
                    elif en_ocl:
                        historial_prendas[ide]["estado"] = "Ocluida..."

                    estado_prendas[ide] = historial_prendas[ide]["estado"]

                    # CSV detalle
                    w_csv.writerow([
                        frame_count, ide, tid,
                        int(x1), int(y1), int(x2), int(y2),
                        cx, cy,
                        int(en_ocl),
                        estado_prendas[ide],
                        int(confirmado),
                    ])

                tracks = np.array([
                    [x1,y1,x2,y2, ide, conf, cls, 0]
                    for (x1,y1,x2,y2,ide,tid,conf,cls,cx,cy) in tracks_est
                ]) if tracks_est else []

        # ── Actualizar estadísticas de resumen ────────────────────────────────
        ids_todos = recovery.get_ids_estables_activos()
        stats.update(
            raw_count    = len(raw_det),
            track_count  = len(tracks) if hasattr(tracks, '__len__') else 0,
            ids_activos  = ids_todos,
            alerta_global= alerta_global,
            estado_prendas = estado_prendas,
        )

        # ── Dibujar y escribir frame ──────────────────────────────────────────
        canvas = dibujar_cajas(canvas, raw_det, tracks, estado_prendas,
                               poly_escaner=poly_escaner, poly_bolsa=poly_bolsa)
        canvas = dibujar_panel(canvas, vid_width, canvas_width, frame_count,
                               len(raw_det),
                               len(tracks) if hasattr(tracks,'__len__') else 0,
                               len(ids_todos))
        out.write(canvas)

    cap.release()
    out.release()
    f_csv.close()
    stats.mark_end()

    elapsed = time.time() - t0
    print(f"\n    ✓ {frame_count} frames | {elapsed:.1f}s | "
          f"{stats.proc_fps:.1f} fps | "
          f"{stats.total_ids_unicos} IDs → {video_out}")

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# ENTRADA PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Detectar todos los videos disponibles ─────────────────────────────────
    patron   = os.path.join(VIDEOS_DIR, "video*.mp4")
    videos   = sorted(glob.glob(patron),
                      key=lambda p: int(''.join(filter(str.isdigit, Path(p).stem)) or '0'))

    if not videos:
        print(f"[ERROR] No se encontraron videos en '{VIDEOS_DIR}/'")
        exit(1)

    print(f"[batch] {len(videos)} videos encontrados:\n  " + "\n  ".join(videos))

    # ── Cargar ROIs del YAML ──────────────────────────────────────────────────
    rois_yaml = cargar_rois_yaml(YAML_ROI)

    # ── Cargar modelos (una sola vez) ─────────────────────────────────────────
    print("\nCargando modelos (una sola vez)...")
    model, tracker = inicializar_modelos()
    print("Modelos listos. Iniciando batch v3...\n")

    # ── Procesar cada video ───────────────────────────────────────────────────
    all_stats = []
    for vp in videos:
        nombre    = Path(vp).stem
        video_out = os.path.join(OUTPUT_DIR, f"{PREFIX}_resultado_{nombre}.mp4")
        csv_out   = os.path.join(OUTPUT_DIR, f"{PREFIX}_resultado_{nombre}.csv")

        poly_escaner, poly_bolsa = obtener_roi_video(vp, rois_yaml)

        print(f"\n▶  {vp}  →  {video_out}")
        stats = procesar_video(
            vp, video_out, csv_out,
            model, tracker,
            poly_escaner, poly_bolsa
        )
        if stats:
            all_stats.append(stats)

    # ── Generar resumen ───────────────────────────────────────────────────────
    if all_stats:
        guardar_csv_resumen(all_stats, CSV_RESUMEN)
        generar_imagen_comparativa(all_stats, IMG_REPORTE)

    print(f"\n✅ Batch v3 completo.")
    print(f"   • Videos procesados : {len(all_stats)}")
    print(f"   • CSV resumen       : {CSV_RESUMEN}")
    print(f"   • Imagen reporte    : {IMG_REPORTE}")
    print(f"   • Resultados en     : {OUTPUT_DIR}/")
