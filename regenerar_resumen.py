"""
regenerar_resumen.py
====================
Lee los CSVs de detalle existentes (v3_resultado_videoN.csv) y el resumen
actual, luego genera un nuevo resumen enriquecido con:
  - Total_Cajas_Rojas   (detecciones brutas RT-DETR por video)
  - Total_Cajas_Verdes  (tracks activos confirmados por video)
  - Avg_Cajas_Rojas_Frame
  - Avg_Cajas_Verdes_Frame

No requiere reprocesar ningún video.
"""

import csv
import os
import glob
from pathlib import Path
from collections import defaultdict

# ── módulo de reporte ─────────────────────────────────────────────────────────
from modules.report import generar_imagen_comparativa

# ──────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR      = "videos_salida"
PREFIX          = "v3"
CSV_RESUMEN_IN  = os.path.join(OUTPUT_DIR, f"{PREFIX}_resumen_batch.csv")
CSV_RESUMEN_OUT = os.path.join(OUTPUT_DIR, f"{PREFIX}_resumen_enriquecido.csv")
IMG_REPORTE_OUT = os.path.join(OUTPUT_DIR, f"{PREFIX}_reporte_comparativo_v2.png")
# ──────────────────────────────────────────────────────────────────────────────


def leer_resumen_base(csv_path):
    """Lee el CSV de resumen existente como lista de dicts."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def calcular_cajas_desde_detalle(csv_detalle_path):
    """
    Analiza el CSV de detalle para obtener:
    - total_cajas_verdes: suma de tracks por frame (cada fila = 1 caja verde)
    - frames con al menos un track
    - ids_unicos confirmados (Confirmado == 1)
    """
    total_cajas_verdes  = 0
    frames_con_tracks   = set()
    ids_confirmados     = set()

    with open(csv_detalle_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_cajas_verdes += 1
            frames_con_tracks.add(int(row["Frame"]))
            if int(row.get("Confirmado", 0)) == 1:
                ids_confirmados.add(int(row["ID_Estable"]))

    return total_cajas_verdes, ids_confirmados


def generar_imagen_desde_resumen(rows_enriquecidas):
    """
    Genera el PNG comparativo usando los datos del resumen enriquecido.
    Construye objetos VideoStats sintéticos a partir de los datos del CSV.
    """
    # Import en línea para evitar dependencia circular
    from modules.report import VideoStats

    stats_list = []
    for r in rows_enriquecidas:
        frames  = int(r["Frames_Procesados"])
        fps_src = int(r["FPS_Fuente"])
        s = VideoStats(r["Video"], fps_src, frames)

        # Simular mark_start / mark_end a partir del FPS de procesamiento
        proc_fps = float(r["FPS_Procesamiento"])
        import time as _time
        s._t_start = 0.0
        s._t_end   = frames / proc_fps if proc_fps > 0 else 0.0

        # Restaurar acumuladores desde el CSV
        s.frames_procesados      = frames
        s.total_detecciones      = int(r["Total_Cajas_Rojas"])
        s.total_tracks           = int(r["Total_Cajas_Verdes"])
        s.ids_estables_set       = set(range(int(r["IDs_Unicos"])))   # set sintético
        s.alertas_frames         = int(r["Frames_Con_Alerta"])
        s.max_tracks_simultaneos = int(r["Max_Tracks_Simultaneos"])
        s.max_dets_simultaneos   = int(r["Max_Detecciones_Simultaneas"])

        # Reconstruir estado_counts desde columnas disponibles
        alerta_f  = int(r["Frames_Con_Alerta"])
        total_f   = frames
        s.estado_counts = {
            "ALERTA: EVASION": alerta_f,
            "Detectando...":   max(0, total_f - alerta_f),
        }
        stats_list.append(s)

    generar_imagen_comparativa(stats_list, IMG_REPORTE_OUT)


# ──────────────────────────────────────────────────────────────────────────────
# CABECERA DEL NUEVO CSV
# ──────────────────────────────────────────────────────────────────────────────

NUEVO_HEADER = [
    "Video",
    "FPS_Fuente",
    "Frames_Procesados",
    "Duracion_seg",
    "IDs_Unicos",
    "IDs_Confirmados",
    "Max_Tracks_Simultaneos",
    "Max_Detecciones_Simultaneas",
    "Total_Cajas_Rojas",        # ← NUEVO
    "Total_Cajas_Verdes",       # ← NUEVO
    "Avg_Cajas_Rojas_Frame",    # ← NUEVO (= Avg_Detecciones_Frame)
    "Avg_Cajas_Verdes_Frame",   # ← NUEVO (= Avg_Tracks_Frame)
    "Frames_Con_Alerta",
    "Pct_Frames_Alerta",
    "FPS_Procesamiento",
]


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(CSV_RESUMEN_IN):
        print(f"[ERROR] No se encontró el resumen base: {CSV_RESUMEN_IN}")
        print("        Ejecuta batch_test.py primero.")
        return

    resumen_base = leer_resumen_base(CSV_RESUMEN_IN)
    print(f"[OK] Resumen base leído: {len(resumen_base)} videos\n")

    rows_out = []

    for r in resumen_base:
        video_name = r["Video"]  # e.g. "video0"
        csv_detalle = os.path.join(OUTPUT_DIR, f"{PREFIX}_resultado_{video_name}.csv")

        # ── Cajas rojas: avg × frames (ya conocido del resumen base) ──────────
        frames     = int(r["Frames_Procesados"])
        avg_det    = float(r["Avg_Detecciones_Frame"])
        total_rojas = round(avg_det * frames)

        # ── Cajas verdes + IDs confirmados: leídos del CSV de detalle ─────────
        if os.path.exists(csv_detalle):
            total_verdes, ids_conf = calcular_cajas_desde_detalle(csv_detalle)
            ids_confirmados = len(ids_conf)
            print(f"  {video_name}: {total_rojas:>6} cajas rojas | "
                  f"{total_verdes:>6} cajas verdes | "
                  f"{ids_confirmados:>3} IDs confirmados")
        else:
            print(f"  {video_name}: CSV detalle no encontrado ({csv_detalle})")
            total_verdes    = 0
            ids_confirmados = int(r["IDs_Unicos"])

        avg_rojas  = round(total_rojas  / frames, 2) if frames else 0
        avg_verdes = round(total_verdes / frames, 2) if frames else 0

        new_row = {
            "Video":                     video_name,
            "FPS_Fuente":                r["FPS_Fuente"],
            "Frames_Procesados":         frames,
            "Duracion_seg":              r["Duracion_seg"],
            "IDs_Unicos":                r["IDs_Unicos"],
            "IDs_Confirmados":           ids_confirmados,
            "Max_Tracks_Simultaneos":    r["Max_Tracks_Simultaneos"],
            "Max_Detecciones_Simultaneas": r["Max_Detecciones_Simultaneas"],
            "Total_Cajas_Rojas":         total_rojas,
            "Total_Cajas_Verdes":        total_verdes,
            "Avg_Cajas_Rojas_Frame":     avg_rojas,
            "Avg_Cajas_Verdes_Frame":    avg_verdes,
            "Frames_Con_Alerta":         r["Frames_Con_Alerta"],
            "Pct_Frames_Alerta":         r["Pct_Frames_Alerta"],
            "FPS_Procesamiento":         r["FPS_Procesamiento"],
        }
        rows_out.append(new_row)

    # ── Escribir CSV enriquecido ──────────────────────────────────────────────
    with open(CSV_RESUMEN_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=NUEVO_HEADER)
        w.writeheader()
        w.writerows(rows_out)

    print(f"\n[OK] CSV enriquecido → {CSV_RESUMEN_OUT}")

    # ── Generar imagen comparativa actualizada ────────────────────────────────
    generar_imagen_desde_resumen(rows_out)
    print(f"[OK] Imagen comparativa → {IMG_REPORTE_OUT}")

    # ── Imprimir tabla resumen ─────────────────────────────────────────────────
    print()
    print(f"{'Video':<10} {'Frames':>7} {'IDs':>5} {'ID_OK':>6} "
          f"{'Rojas':>8} {'Verdes':>8} {'Avg_R':>6} {'Avg_V':>6} {'%Alert':>7}")
    print("─" * 75)
    for r in rows_out:
        print(f"{r['Video']:<10} {r['Frames_Procesados']:>7} "
              f"{r['IDs_Unicos']:>5} {r['IDs_Confirmados']:>6} "
              f"{r['Total_Cajas_Rojas']:>8} {r['Total_Cajas_Verdes']:>8} "
              f"{r['Avg_Cajas_Rojas_Frame']:>6} {r['Avg_Cajas_Verdes_Frame']:>6} "
              f"{r['Pct_Frames_Alerta']:>7}%")

    print(f"\n✅ Listo. Archivos en {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
