import cv2
import numpy as np
import config


# ──────────────────────────────────────────────────────────────────────────────
# LIENZO
# ──────────────────────────────────────────────────────────────────────────────

def crear_lienzo(frame, vid_width, vid_height, panel_width=400):
    canvas_width = vid_width + panel_width
    canvas = np.zeros((vid_height, canvas_width, 3), dtype=np.uint8)
    canvas[:, :vid_width] = frame
    return canvas, canvas_width, vid_width


# ──────────────────────────────────────────────────────────────────────────────
# ZONAS
# ──────────────────────────────────────────────────────────────────────────────

def dibujar_zonas(canvas, poly_escaner=None, poly_bolsa=None):
    if poly_escaner is None:
        poly_escaner = np.array(config.ZONA_ESCANER, np.int32)
    if poly_bolsa is None:
        poly_bolsa = np.array(config.ZONA_BOLSA, np.int32)

    for pts, color_fill, color_line, label in [
        (poly_escaner, (180, 160, 0), (255, 220, 0), "ZONA ESCAN"),
        (poly_bolsa,   (160, 0, 160), (255, 0, 255), "ZONA BOLSA"),
    ]:
        pts_r = pts.reshape((-1, 1, 2))
        ov = canvas.copy()
        cv2.fillPoly(ov, [pts_r], color_fill)
        cv2.addWeighted(ov, 0.10, canvas, 0.90, 0, canvas)
        cv2.polylines(canvas, [pts_r], True, color_line, 2)
        lx, ly = int(pts[0][0]), max(12, int(pts[0][1]) - 6)
        cv2.putText(canvas, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(canvas, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_line, 1)
    return canvas


# ──────────────────────────────────────────────────────────────────────────────
# CAJAS DE TRACKING (solo tracks confirmados, sin cajas rojas ruidosas)
# ──────────────────────────────────────────────────────────────────────────────

def dibujar_cajas(canvas, raw_det, tracks, estado_prendas=None,
                  ids_confirmados=None, poly_escaner=None, poly_bolsa=None):
    if estado_prendas is None:
        estado_prendas = {}
    if ids_confirmados is None:
        ids_confirmados = set()

    canvas = dibujar_zonas(canvas, poly_escaner, poly_bolsa)

    alerta = False

    for t in tracks:
        x1, y1, x2, y2, track_id, conf, cls, ind = t
        track_id = int(track_id)

        # Ocultar IDs no confirmados (ruido)
        if ids_confirmados and track_id not in ids_confirmados:
            continue

        estado_actual = estado_prendas.get(track_id, "")

        # Color por estado
        if "ALERTA" in estado_actual:
            color = (0, 0, 255);   alerta = True
        elif "Escaneando" in estado_actual:
            color = (0, 200, 255)
        elif "Desalarmado" in estado_actual:
            color = (0, 255, 128)
        elif "Ocluida" in estado_actual:
            color = (180, 100, 255)
        else:
            color = (0, 210, 0)

        # Caja con sombra
        cv2.rectangle(canvas, (int(x1)+2, int(y1)+2), (int(x2)+2, int(y2)+2),
                      (0, 0, 0), 2)
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

        # Etiqueta ID
        lbl = f"ID:{track_id}"
        for th, cl in [(4, (0,0,0)), (2, color)]:
            cv2.putText(canvas, lbl, (int(x1), int(y1) - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, cl, th)

        # Estado debajo del ID
        if estado_actual:
            for th, cl in [(3, (0,0,0)), (1, color)]:
                cv2.putText(canvas, estado_actual, (int(x1), int(y1) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, cl, th)

        # Centroide
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        cv2.circle(canvas, (cx, cy), 5, (0, 255, 255), -1)
        cv2.circle(canvas, (cx, cy), 5, (0, 0, 0), 1)

    # Banner alerta
    if alerta:
        h, w = canvas.shape[:2]
        ov = canvas.copy()
        cv2.rectangle(ov, (0, 0), (w, 52), (0, 0, 160), -1)
        cv2.addWeighted(ov, 0.6, canvas, 0.4, 0, canvas)
        cv2.putText(canvas, "!!!  POSIBLE EVASION DETECTADA  !!!",
                    (28, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(canvas, "!!!  POSIBLE EVASION DETECTADA  !!!",
                    (28, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

    return canvas


# ──────────────────────────────────────────────────────────────────────────────
# PANEL LATERAL — limpio y con conteo de prendas reales
# ──────────────────────────────────────────────────────────────────────────────

def dibujar_panel(canvas, vid_width, canvas_width, frame_count,
                  raw_count, track_count, total_ids,
                  prendas_reales=None):
    x  = vid_width + 16
    bg = (14, 18, 26)
    cv2.rectangle(canvas, (vid_width, 0), (canvas_width, canvas.shape[0]), bg, -1)

    # Título
    cv2.putText(canvas, "TRACKING AUDIT",
                (x, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (240, 240, 255), 2)
    cv2.line(canvas, (x, 55), (canvas_width - 14, 55), (50, 60, 80), 1)

    # Tracker activo
    cv2.putText(canvas, f"Tracker: {config.TRACKER_TYPE.upper()}",
                (x, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (120, 180, 255), 1)

    # Frame
    cv2.putText(canvas, f"Frame: {frame_count}",
                (x, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (160, 160, 185), 1)

    # Tracks confirmados (verde)
    cv2.putText(canvas, f"Tracks activos: {track_count}",
                (x, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 210, 80), 2)

    # IDs únicos
    cv2.putText(canvas, f"IDs unicos (raw): {total_ids}",
                (x, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 160), 1)

    # ── PRENDAS REALES (la métrica clave) ────────────────────────────────────
    if prendas_reales is not None:
        label_pr  = f"PRENDAS: {prendas_reales}"
        color_pr  = (0, 255, 200)
        cv2.rectangle(canvas, (x - 4, 200), (canvas_width - 14, 240), (20, 40, 40), -1)
        cv2.rectangle(canvas, (x - 4, 200), (canvas_width - 14, 240), (0, 180, 140), 1)
        cv2.putText(canvas, label_pr,
                    (x + 2, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4)
        cv2.putText(canvas, label_pr,
                    (x + 2, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color_pr, 2)

    # Línea separadora
    cv2.line(canvas, (x, 255), (canvas_width - 14, 255), (40, 50, 65), 1)

    # Parámetros
    cv2.putText(canvas, "Config:",
                (x, 278), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 210), 1)
    params = [
        f"  conf min : {config.DET_CONFIDENCE}",
        f"  fp16 GPU : {config.USE_FP16}",
        f"  stab min : {config.TRACK_MIN_STABLE_FRAMES}f",
        f"  cons gap : {config.CONSOLIDATION_MAX_GAP}f",
        f"  cons sim : {config.CONSOLIDATION_SIM_THRESH}",
    ]
    for i, txt in enumerate(params):
        cv2.putText(canvas, txt, (x, 300 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (130, 140, 165), 1)

    return canvas


# ──────────────────────────────────────────────────────────────────────────────
# FRAME RESUMEN FINAL (se agrega al final del video)
# ──────────────────────────────────────────────────────────────────────────────

def crear_frame_resumen(vid_width, vid_height, panel_width,
                        video_name: str,
                        prendas_reales: int,
                        resumen_prendas: list,
                        total_ids: int,
                        proc_fps: float,
                        duracion_seg: float):
    """
    Genera un frame de resumen final de fondo oscuro con los datos clave.
    Se agrega N veces al final del video para que sea visible unos segundos.
    """
    w = vid_width + panel_width
    h = vid_height
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (12, 15, 22)

    # ── Título ────────────────────────────────────────────────────────────────
    cv2.putText(canvas, "RESUMEN FINAL DEL VIDEO",
                (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(canvas, video_name,
                (40, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (120, 160, 220), 1)
    cv2.line(canvas, (40, 108), (w - 40, 108), (50, 60, 80), 1)

    # ── Métrica principal: PRENDAS DETECTADAS ─────────────────────────────────
    big_label = f"{prendas_reales}"
    cv2.putText(canvas, big_label,
                (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 255, 200), 8)
    cv2.putText(canvas, "PRENDAS EN ESCENA",
                (100, 265), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 200, 160), 2)

    # ── Detalle por prenda ────────────────────────────────────────────────────
    cv2.line(canvas, (40, 290), (w // 2 - 20, 290), (50, 60, 80), 1)
    cv2.putText(canvas, "Detalle por prenda:",
                (40, 314), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 210), 1)

    for i, p in enumerate(resumen_prendas[:10]):   # máx 10
        txt = (f"  Prenda {p['prenda']:>2}: "
               f"frames {p['first_frame']:>5}→{p['last_frame']:>5} | "
               f"IDs fusionados: {len(p['ids'])}")
        cv2.putText(canvas, txt,
                    (40, 338 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 200, 140), 1)

    # ── Stats secundarias (columna derecha) ───────────────────────────────────
    rx = w // 2 + 40
    cv2.line(canvas, (rx - 20, 290), (w - 40, 290), (50, 60, 80), 1)
    stats = [
        f"IDs raw del tracker : {total_ids}",
        f"Duracion del video  : {duracion_seg:.1f} seg",
        f"FPS de proceso      : {proc_fps:.1f} fps",
        f"Tracker usado       : {config.TRACKER_TYPE.upper()}",
        f"Detector            : RT-DETR",
    ]
    for i, txt in enumerate(stats):
        cv2.putText(canvas, txt, (rx, 314 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 170, 200), 1)

    # ── Footer ────────────────────────────────────────────────────────────────
    cv2.line(canvas, (40, h - 40), (w - 40, h - 40), (50, 60, 80), 1)
    cv2.putText(canvas, "Sistema de Auditoria de Tracking — v4",
                (40, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 90, 110), 1)

    return canvas