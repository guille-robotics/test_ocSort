import cv2
import numpy as np
import config

def crear_lienzo(frame, vid_width, vid_height, panel_width=400):
    canvas_width = vid_width + panel_width
    canvas = np.zeros((vid_height, canvas_width, 3), dtype=np.uint8)
    canvas[:, :vid_width] = frame
    return canvas, canvas_width, vid_width

def dibujar_zonas(canvas):
    # Dibujar Zona Escáner (Azul claro)
    pts_escaner = np.array(config.ZONA_ESCANER, np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts_escaner], isClosed=True, color=(255, 200, 0), thickness=2)
    cv2.putText(canvas, "ZONA ESCANER", (config.ZONA_ESCANER[0][0], config.ZONA_ESCANER[0][1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 2)

    # Dibujar Zona Bolsa (Morado)
    pts_bolsa = np.array(config.ZONA_BOLSA, np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts_bolsa], isClosed=True, color=(255, 0, 255), thickness=2)
    cv2.putText(canvas, "ZONA BOLSA", (config.ZONA_BOLSA[0][0], config.ZONA_BOLSA[0][1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    return canvas

def dibujar_cajas(canvas, raw_det, tracks, estado_prendas=None):
    if estado_prendas is None: estado_prendas = {}
    
    canvas = dibujar_zonas(canvas)

    # Dibujar detecciones crudas (Rojo delgado)
    for d in raw_det:
        rx1, ry1, rx2, ry2, rconf, rcls = d
        cv2.rectangle(canvas, (int(rx1), int(ry1)), (int(rx2), int(ry2)), (0, 0, 255), 1)

    alerta_en_pantalla = False

    # Dibujar tracks oficiales
    for t in tracks:
        x1, y1, x2, y2, track_id, conf, cls, ind = t
        track_id = int(track_id)
        
        # Lógica de colores según el estado
        color_caja = (0, 255, 0) # Verde por defecto
        estado_actual = estado_prendas.get(track_id, "")
        
        if "ALERTA" in estado_actual:
            color_caja = (0, 0, 255) # Rojo brillante si hay evasión
            alerta_en_pantalla = True
            
        # Dibujar caja y texto
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color_caja, 3)
        cv2.putText(canvas, f"ID: {track_id} - {estado_actual}", (int(x1), int(y1) - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_caja, 2)
        
        # Dibujar el centroide matemático
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(canvas, (cx, cy), 5, (0, 255, 255), -1)

    # Dibujar el letrero gigante si alguien se saltó el escáner
    if alerta_en_pantalla:
        cv2.putText(canvas, "!!! POSIBLE EVASION DETECTADA !!!", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
    return canvas

def dibujar_panel(canvas, vid_width, canvas_width, frame_count, raw_count, track_count, total_ids):
    x_panel = vid_width + 20
    
    # Título
    cv2.putText(canvas, "AUDITORIA DE TRACKING", (x_panel, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.line(canvas, (x_panel, 65), (canvas_width - 20, 65), (255, 255, 255), 1)
    
    # Estadísticas dinámicas
    cv2.putText(canvas, f"Frame: {frame_count}", (x_panel, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    cv2.putText(canvas, f"Detecciones (RT-DETR): {raw_count}", (x_panel, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(canvas, "(Cajas Rojas)", (x_panel, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.putText(canvas, f"Tracks Activos: {track_count}", (x_panel, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(canvas, "(Cajas Verdes)", (x_panel, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.putText(canvas, f"Total IDs Unicos: {total_ids}", (x_panel, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Parámetros estáticos
    cv2.putText(canvas, f"Parametros Activos:", (x_panel, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(canvas, f"- min_hits: {config.TRACK_MIN_HITS}", (x_panel, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(canvas, f"- max_age: {config.TRACK_MAX_AGE}", (x_panel, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(canvas, f"- Confianza min: {config.DET_CONFIDENCE}", (x_panel, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return canvas