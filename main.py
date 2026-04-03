import cv2
import os
import numpy as np
import math
import csv
import config
from modules.vision import inicializar_modelos
from modules.ui import crear_lienzo, dibujar_cajas, dibujar_panel


# ==============================================================================
# SISTEMA DE RECUPERACIÓN DE ID
# Cuando OC-SORT pierde un track por movimiento brusco y asigna un ID nuevo,
# esta capa lo detecta y lo remapea al ID original correcto.
# ==============================================================================

class IdRecoverySystem:
    """
    Mantiene un 'mapa de traducción' entre IDs internos de OC-SORT (que pueden
    cambiar cuando hay saltos) y los IDs estables que se muestran al usuario.

    Estrategia:
    1. Guarda la última posición conocida de cada ID ESTABLE.
    2. Cuando aparece un ID NUEVO del tracker, calcula su distancia a todos los
       IDs perdidos recientemente.
    3. Si la distancia es menor a ID_RECOVERY_MAX_DIST, reasigna el ID perdido
       al ID nuevo (recuperacion de identidad).
    """
    def __init__(self):
        self.id_map = {}          # tracker_id -> id_estable
        self.id_inverso = {}      # id_estable -> tracker_id activo actual
        self.ultima_pos = {}      # id_estable -> (cx, cy)
        self.ultimo_frame = {}    # id_estable -> frame en que fue visto por ultima vez
        self.siguiente_id = 1     # Proximo ID estable a asignar

    def get_id_estable(self, tracker_id, cx, cy, frame_count):
        """Dada un tracker_id (que puede cambiar) devuelve el ID estable."""

        # Si ya conocemos este tracker_id, solo actualizamos su posicion
        if tracker_id in self.id_map:
            id_estable = self.id_map[tracker_id]
            self.ultima_pos[id_estable] = (cx, cy)
            self.ultimo_frame[id_estable] = frame_count
            return id_estable

        # --- Es un tracker_id NUEVO ---
        # Buscar si coincide con algun ID estable perdido recientemente
        mejor_id = None
        mejor_dist = float('inf')

        for id_est, (ox, oy) in self.ultima_pos.items():
            # Solo considerar IDs que no esten activos en este momento
            if id_est in self.id_inverso and self.id_inverso[id_est] in self.id_map:
                continue  # Este ID estable ya tiene un tracker_id activo, saltar

            # Solo recuperar si no lleva demasiados frames perdido
            frames_perdido = frame_count - self.ultimo_frame.get(id_est, 0)
            if frames_perdido > config.ID_RECOVERY_MAX_AGE:
                continue

            dist = math.hypot(cx - ox, cy - oy)
            if dist < mejor_dist:
                mejor_dist = dist
                mejor_id = id_est

        if mejor_id is not None and mejor_dist < config.ID_RECOVERY_MAX_DIST:
            # Recuperacion exitosa: el tracker_id nuevo es el mismo objeto que mejor_id
            id_estable = mejor_id
        else:
            # Es genuinamente un objeto nuevo
            id_estable = self.siguiente_id
            self.siguiente_id += 1

        # Registrar la asociacion tracker_id <-> id_estable
        self.id_map[tracker_id] = id_estable
        self.id_inverso[id_estable] = tracker_id
        self.ultima_pos[id_estable] = (cx, cy)
        self.ultimo_frame[id_estable] = frame_count
        return id_estable

    def get_ids_estables_activos(self):
        return set(self.id_map.values())


def main():
    os.makedirs(config.CROPS_DIR, exist_ok=True)
    model, tracker = inicializar_modelos()

    cap = cv2.VideoCapture(config.VIDEO_IN)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    panel_width = 400
    canvas_width = vid_width + panel_width
    out = cv2.VideoWriter(config.VIDEO_OUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (canvas_width, vid_height))

    # CSV
    os.makedirs(os.path.dirname(config.CSV_OUT) or '.', exist_ok=True)
    f_csv = open(config.CSV_OUT, mode='w', newline='', encoding='utf-8')
    csv_writer = csv.writer(f_csv)
    csv_writer.writerow(['Frame', 'ID_Estable', 'ID_Tracker', 'X1', 'Y1', 'X2', 'Y2', 'CX', 'CY', 'Estado'])

    # Sistemas de estado
    recovery = IdRecoverySystem()
    historial_prendas = {}
    estado_prendas = {}
    ids_guardados = set()
    frame_count = 0

    poly_escaner = np.array(config.ZONA_ESCANER, np.int32)
    poly_bolsa = np.array(config.ZONA_BOLSA, np.int32)

    print("Procesando video con ID Recovery + GIoU Tracking...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        canvas, _, _ = crear_lienzo(frame, vid_width, vid_height, panel_width)
        alerta_global = False

        # 1. Deteccion con RT-DETR
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
        tracks = []

        if len(results[0].boxes) > 0:
            cajas_crudas = results[0].boxes.data.cpu().numpy()

            # 2. NMS Manual para eliminar duplicados del RT-DETR
            boxes_nms = []
            confs_nms = []
            for caja in cajas_crudas:
                rx1, ry1, rx2, ry2, rconf, rcls = caja
                boxes_nms.append([int(rx1), int(ry1), int(rx2 - rx1), int(ry2 - ry1)])
                confs_nms.append(float(rconf))

            indices = cv2.dnn.NMSBoxes(boxes_nms, confs_nms,
                                       score_threshold=config.DET_CONFIDENCE,
                                       nms_threshold=0.3)

            # 3. Filtro de area
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

                # Construir lista de tracks con IDs ESTABLES via recuperacion
                tracks_estables = []
                for t in tracks_raw:
                    x1, y1, x2, y2, tracker_id, conf, cls, ind = t
                    tracker_id = int(tracker_id)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    id_estable = recovery.get_id_estable(tracker_id, cx, cy, frame_count)
                    tracks_estables.append((x1, y1, x2, y2, id_estable, tracker_id, conf, cls, cx, cy))

                for (x1, y1, x2, y2, id_estable, tracker_id, conf, cls, cx, cy) in tracks_estables:

                    # 4. Inicializar o actualizar historial usando el ID ESTABLE
                    if id_estable not in historial_prendas:
                        historial_prendas[id_estable] = {
                            "paso_escaner": False,
                            "estado": "Detectando...",
                            "frames_sospechosos": 0,
                            "ultima_pos": (cx, cy)
                        }
                    else:
                        historial_prendas[id_estable]["ultima_pos"] = (cx, cy)

                    # 5. Logica de zonas
                    en_escaner = cv2.pointPolygonTest(poly_escaner, (cx, cy), False) >= 0
                    en_bolsa = cv2.pointPolygonTest(poly_bolsa, (cx, cy), False) >= 0

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

                    # 6. Guardar en CSV
                    csv_writer.writerow([frame_count, id_estable, tracker_id,
                                         int(x1), int(y1), int(x2), int(y2),
                                         cx, cy, estado_prendas[id_estable]])

                    # 7. Guardar primer recorte
                    if id_estable not in ids_guardados:
                        ix1 = max(0, int(x1))
                        iy1 = max(0, int(y1))
                        ix2 = min(vid_width, int(x2))
                        iy2 = min(vid_height, int(y2))
                        crop = frame[iy1:iy2, ix1:ix2]
                        if crop.size > 0:
                            cv2.imwrite(os.path.join(config.CROPS_DIR, f"ropa_id_{id_estable}.jpg"), crop)
                            ids_guardados.add(id_estable)

                # Reconstruir tracks con IDs estables para la UI
                tracks = np.array([
                    [x1, y1, x2, y2, id_estable, conf, cls, 0]
                    for (x1, y1, x2, y2, id_estable, tracker_id, conf, cls, cx, cy) in tracks_estables
                ]) if tracks_estables else []

        ids_unicos = recovery.get_ids_estables_activos()

        # UI
        canvas = dibujar_cajas(canvas, raw_det, tracks, estado_prendas)
        canvas = dibujar_panel(canvas, vid_width, canvas_width, frame_count,
                               len(raw_det), len(tracks) if len(tracks) > 0 else 0,
                               len(ids_unicos))
        out.write(canvas)

    cap.release()
    out.release()
    f_csv.close()
    print("Auditoria finalizada con ID Recovery!")


if __name__ == "__main__":
    main()