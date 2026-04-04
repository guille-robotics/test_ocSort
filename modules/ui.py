import cv2
import numpy as np
import config


def crear_lienzo(frame, vid_width, vid_height, panel_width=400):
    canvas_width = vid_width + panel_width
    canvas = np.zeros((vid_height, canvas_width, 3), dtype=np.uint8)
    canvas[:, :vid_width] = frame
    return canvas, canvas_width, vid_width


def dibujar_zonas(canvas, poly_escaner=None, poly_bolsa=None):
    """
    Dibuja los polígonos de zona escáner y zona bolsa.

    Si se pasan poly_escaner / poly_bolsa como np.array int32 se usan esos
    (provenientes del YAML por video). Si son None se recae en config.py.
    """
    if poly_escaner is None:
        poly_escaner = np.array(config.ZONA_ESCANER, np.int32)
    if poly_bolsa is None:
        poly_bolsa = np.array(config.ZONA_BOLSA, np.int32)

    # Zona Escáner — azul cian semitransparente
    pts_esc = poly_escaner.reshape((-1, 1, 2))
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts_esc], (200, 180, 0))
    cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0, canvas)
    cv2.polylines(canvas, [pts_esc], isClosed=True, color=(255, 220, 0), thickness=2)
    lbl_esc = tuple(poly_escaner[0])
    cv2.putText(canvas, "ZONA ESCANER",
                (lbl_esc[0], max(0, lbl_esc[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 2)

    # Zona Bolsa — magenta semitransparente
    pts_bol = poly_bolsa.reshape((-1, 1, 2))
    overlay2 = canvas.copy()
    cv2.fillPoly(overlay2, [pts_bol], (200, 0, 200))
    cv2.addWeighted(overlay2, 0.12, canvas, 0.88, 0, canvas)
    cv2.polylines(canvas, [pts_bol], isClosed=True, color=(255, 0, 255), thickness=2)
    lbl_bol = tuple(poly_bolsa[0])
    cv2.putText(canvas, "ZONA BOLSA",
                (lbl_bol[0], max(0, lbl_bol[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

    return canvas


def dibujar_cajas(canvas, raw_det, tracks, estado_prendas=None,
                  poly_escaner=None, poly_bolsa=None):
    if estado_prendas is None:
        estado_prendas = {}

    canvas = dibujar_zonas(canvas, poly_escaner, poly_bolsa)

    # Detecciones crudas — borde rojo delgado con transparencia
    for d in raw_det:
        rx1, ry1, rx2, ry2, rconf, rcls = d
        cv2.rectangle(canvas,
                      (int(rx1), int(ry1)), (int(rx2), int(ry2)),
                      (0, 50, 220), 1)

    alerta_en_pantalla = False

    for t in tracks:
        x1, y1, x2, y2, track_id, conf, cls, ind = t
        track_id = int(track_id)
        estado_actual = estado_prendas.get(track_id, "")

        # Color de caja según estado
        if "ALERTA" in estado_actual:
            color_caja = (0, 0, 255)
            alerta_en_pantalla = True
        elif "Escaneando" in estado_actual:
            color_caja = (0, 200, 255)
        elif "Desalarmado" in estado_actual:
            color_caja = (0, 255, 128)
        elif "Ocluida" in estado_actual:
            color_caja = (180, 100, 255)
        else:
            color_caja = (0, 220, 0)

        # Sombra sutil para que la caja resalte sobre cualquier fondo
        cv2.rectangle(canvas,
                      (int(x1)+2, int(y1)+2), (int(x2)+2, int(y2)+2),
                      (0, 0, 0), 2)
        cv2.rectangle(canvas,
                      (int(x1), int(y1)), (int(x2), int(y2)),
                      color_caja, 3)

        label = f"ID:{track_id}"
        cv2.putText(canvas, label,
                    (int(x1), int(y1) - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)
        cv2.putText(canvas, label,
                    (int(x1), int(y1) - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_caja, 2)

        # Estado debajo del label
        if estado_actual:
            cv2.putText(canvas, estado_actual,
                        (int(x1), int(y1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(canvas, estado_actual,
                        (int(x1), int(y1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_caja, 1)

        # Centroide
        cx, cy = int((x1+x2)/2), int((y1+y2)/2)
        cv2.circle(canvas, (cx, cy), 5, (0, 255, 255), -1)
        cv2.circle(canvas, (cx, cy), 5, (0, 0, 0), 1)

    # Banner de alerta
    if alerta_en_pantalla:
        h, w = canvas.shape[:2]
        banner_h = 55
        overlay_b = canvas.copy()
        cv2.rectangle(overlay_b, (0, 0), (w, banner_h), (0, 0, 180), -1)
        cv2.addWeighted(overlay_b, 0.6, canvas, 0.4, 0, canvas)
        cv2.putText(canvas, "!!!  POSIBLE EVASION DETECTADA  !!!",
                    (30, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        cv2.putText(canvas, "!!!  POSIBLE EVASION DETECTADA  !!!",
                    (30, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 1)

    return canvas


def dibujar_panel(canvas, vid_width, canvas_width, frame_count,
                  raw_count, track_count, total_ids):
    x  = vid_width + 18
    bg = (16, 20, 28)

    # Fondo del panel
    cv2.rectangle(canvas, (vid_width, 0), (canvas_width, canvas.shape[0]), bg, -1)

    # Título
    cv2.putText(canvas, "TRACKING AUDIT",
                (x, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.line(canvas, (x, 58), (canvas_width - 18, 58), (60, 70, 90), 1)

    # --- Frame ---
    cv2.putText(canvas, f"Frame  {frame_count}",
                (x, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 180), 1)

    # --- Detecciones (rojo) ---
    _barra(canvas, x, 120, canvas_width - 18, raw_count, 10,
           (0, 60, 200), (0, 80, 255), f"Det. RTDETR: {raw_count}")

    # --- Tracks (verde) ---
    _barra(canvas, x, 185, canvas_width - 18, track_count, 10,
           (0, 100, 40), (0, 210, 80), f"Tracks activos: {track_count}")

    # --- Total IDs únicos (amarillo) ---
    cv2.putText(canvas, f"IDs unicos vistos: {total_ids}",
                (x, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 230, 40), 2)

    # --- Parámetros clave ---
    cv2.line(canvas, (x, 295), (canvas_width - 18, 295), (40, 50, 65), 1)
    cv2.putText(canvas, "Parametros activos:",
                (x, 318), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 220), 1)

    params = [
        f"  min_hits : {config.TRACK_MIN_HITS}",
        f"  max_age  : {config.TRACK_MAX_AGE}",
        f"  confianza: {config.DET_CONFIDENCE}",
        f"  fp16     : {config.USE_FP16}",
        f"  estable  : {config.TRACK_MIN_STABLE_FRAMES}f",
    ]
    for i, txt in enumerate(params):
        cv2.putText(canvas, txt,
                    (x, 342 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 160, 180), 1)

    return canvas


def _barra(canvas, x1, y, x2, valor, max_val, color_bg, color_fg, label):
    """Dibuja una mini barra de progreso con etiqueta."""
    bar_h = 18
    total_w = x2 - x1
    fill_w  = int(min(1.0, valor / max_val) * total_w) if max_val else 0

    # Fondo
    cv2.rectangle(canvas, (x1, y), (x2, y + bar_h), color_bg, -1)
    # Relleno
    if fill_w > 0:
        cv2.rectangle(canvas, (x1, y), (x1 + fill_w, y + bar_h), color_fg, -1)
    # Borde
    cv2.rectangle(canvas, (x1, y), (x2, y + bar_h), (60, 70, 90), 1)
    # Texto
    cv2.putText(canvas, label,
                (x1 + 4, y + bar_h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)