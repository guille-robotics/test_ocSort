"""
modules/report.py
=================
Genera:
  1. CSV de resumen  (una fila por video)
  2. Imagen PNG con gráficos comparativos para poder evaluar y comparar modelos
"""

import csv
import os
import numpy as np

# ── intento importar matplotlib ────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")           # sin GUI
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[report] matplotlib no disponible – solo se generará el CSV de resumen.")


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _safe_div(a, b, default=0.0):
    return a / b if b else default


# ──────────────────────────────────────────────────────────────────────────────
# CLASE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

class VideoStats:
    """Acumula métricas frame a frame durante el procesamiento de un video."""

    def __init__(self, video_name: str, source_fps: int, total_frames: int):
        self.video_name    = video_name
        self.source_fps    = source_fps           # FPS original del video
        self.total_frames  = total_frames

        # Contadores acumulados
        self.frames_procesados  = 0
        self.total_detecciones  = 0               # suma de cajas rojas por frame
        self.total_tracks       = 0               # suma de tracks activos por frame
        self.ids_estables_set   = set()           # IDs únicos vistos
        self.alertas_frames     = 0               # frames con ALERTA activa
        self.prendas_reales     = 0               # prendas consolidadas al final

        # Para calcular FPS de procesamiento real
        self._t_start  = None
        self._t_end    = None

        # Distribución de estados (por aparición en CSV)
        self.estado_counts = {}

        # Máx simultáneos (pico)
        self.max_tracks_simultaneos = 0
        self.max_dets_simultaneos   = 0

    # ── llamar una vez al iniciar el loop ─────────────────────────────────────
    def mark_start(self):
        import time
        self._t_start = time.time()

    # ── llamar una vez al terminar el loop ────────────────────────────────────
    def mark_end(self):
        import time
        self._t_end = time.time()

    # ── llamar una vez por frame ──────────────────────────────────────────────
    def update(self, raw_count: int, track_count: int,
               ids_activos: set, alerta_global: bool,
               estado_prendas: dict):
        self.frames_procesados += 1
        self.total_detecciones += raw_count
        self.total_tracks      += track_count
        self.ids_estables_set  |= ids_activos

        if alerta_global:
            self.alertas_frames += 1

        if track_count > self.max_tracks_simultaneos:
            self.max_tracks_simultaneos = track_count
        if raw_count > self.max_dets_simultaneos:
            self.max_dets_simultaneos = raw_count

        for estado in estado_prendas.values():
            self.estado_counts[estado] = self.estado_counts.get(estado, 0) + 1

    # ── propiedades derivadas ────────────────────────────────────────────────
    @property
    def proc_fps(self):
        if self._t_start and self._t_end and (self._t_end - self._t_start) > 0:
            return self.frames_procesados / (self._t_end - self._t_start)
        return 0.0

    @property
    def avg_detecciones(self):
        return _safe_div(self.total_detecciones, self.frames_procesados)

    @property
    def avg_tracks(self):
        return _safe_div(self.total_tracks, self.frames_procesados)

    @property
    def total_ids_unicos(self):
        return len(self.ids_estables_set)

    @property
    def duracion_seg(self):
        return _safe_div(self.frames_procesados, self.source_fps)

    @property
    def alertas_pct(self):
        return _safe_div(self.alertas_frames * 100, self.frames_procesados)

    def to_row(self):
        """Devuelve una lista ordenada para escribir en el CSV de resumen."""
        return [
            self.video_name,
            self.source_fps,
            self.frames_procesados,
            round(self.duracion_seg, 1),
            self.prendas_reales,          # ← PRENDAS REALES (consolidado)
            self.total_ids_unicos,
            self.max_tracks_simultaneos,
            self.max_dets_simultaneos,
            round(self.avg_detecciones, 2),
            round(self.avg_tracks, 2),
            self.alertas_frames,
            round(self.alertas_pct, 1),
            round(self.proc_fps, 1),
        ]


CSV_SUMMARY_HEADER = [
    "Video",
    "FPS_Fuente",
    "Frames_Procesados",
    "Duracion_seg",
    "Prendas_Reales",              # ← NUEVO: conteo consolidado
    "IDs_Unicos",
    "Max_Tracks_Simultaneos",
    "Max_Detecciones_Simultaneas",
    "Avg_Detecciones_Frame",
    "Avg_Tracks_Frame",
    "Frames_Con_Alerta",
    "Pct_Frames_Alerta",
    "FPS_Procesamiento",
]


# ──────────────────────────────────────────────────────────────────────────────
# ESCRITURA DE CSV
# ──────────────────────────────────────────────────────────────────────────────

def guardar_csv_resumen(stats_list: list, csv_path: str):
    """Escribe el CSV resumen con una fila por video."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_SUMMARY_HEADER)
        for s in stats_list:
            w.writerow(s.to_row())
    print(f"[report] CSV resumen → {csv_path}")


# ──────────────────────────────────────────────────────────────────────────────
# IMAGEN COMPARATIVA
# ──────────────────────────────────────────────────────────────────────────────

_ESTADO_COLORS = {
    "ALERTA: EVASION": "#FF4444",
    "Escaneando...":   "#44AAFF",
    "Desalarmado OK":  "#44FF88",
    "Evaluando...":    "#FFB844",
    "Ocluida...":      "#BB88FF",
    "Detectando...":   "#AAAAAA",
}


def generar_imagen_comparativa(stats_list: list, img_path: str):
    """
    Genera un PNG con 6 subgráficos comparativos entre todos los videos.
    """
    if not HAS_MPL:
        print("[report] matplotlib no disponible → imagen no generada.")
        return
    if not stats_list:
        return

    nombres  = [s.video_name for s in stats_list]
    n        = len(nombres)
    x        = np.arange(n)
    bar_w    = 0.6

    # ── Estilo oscuro premium ─────────────────────────────────────────────────
    plt.style.use("dark_background")
    BG   = "#0D1117"
    FG   = "#E6EDF3"
    ACC1 = "#58A6FF"  # azul
    ACC2 = "#3FB950"  # verde
    ACC3 = "#F78166"  # rojo
    ACC4 = "#D2A8FF"  # violeta
    ACC5 = "#FFA657"  # naranja
    ACC6 = "#79C0FF"  # azul claro

    fig = plt.figure(figsize=(max(14, n * 1.4), 22), facecolor=BG)
    fig.suptitle(
        "🎯  Reporte Comparativo de Tracking — v4",
        fontsize=18, fontweight="bold", color=FG, y=0.98
    )

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.55, wspace=0.35,
                           left=0.07, right=0.97,
                           top=0.95, bottom=0.05)

    ax_prendas = fig.add_subplot(gs[0, :])   # fila completa — LA MÉTRICA CLAVE
    ax_ids     = fig.add_subplot(gs[1, 0])
    ax_fps     = fig.add_subplot(gs[1, 1])
    ax_dets    = fig.add_subplot(gs[2, 0])
    ax_tracks  = fig.add_subplot(gs[2, 1])
    ax_alerta  = fig.add_subplot(gs[3, 0])
    ax_est     = fig.add_subplot(gs[3, 1])

    def _style_ax(ax, title, ylabel=""):
        ax.set_facecolor("#161B22")
        ax.set_title(title, color=FG, fontsize=11, pad=8, fontweight="bold")
        ax.set_ylabel(ylabel, color=FG, fontsize=9)
        ax.tick_params(colors=FG, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")
        ax.yaxis.label.set_color(FG)
        ax.grid(axis="y", color="#21262D", linewidth=0.7, linestyle="--", alpha=0.8)

    def _xticks(ax):
        ax.set_xticks(x)
        ax.set_xticklabels(nombres, rotation=40, ha="right", fontsize=8, color=FG)

    # 0 ── PRENDAS REALES — gráfico principal (fila entera) ──────────────────
    pr_vals = [s.prendas_reales for s in stats_list]
    colors_pr = [ACC2 if v > 0 else "#444444" for v in pr_vals]
    bars_pr = ax_prendas.bar(x, pr_vals, bar_w * 0.7, color=colors_pr, alpha=0.90)
    for bar, v in zip(bars_pr, pr_vals):
        ax_prendas.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        str(v), ha="center", va="bottom", color=FG,
                        fontsize=12, fontweight="bold")
    _style_ax(ax_prendas, "🧺  Prendas Reales Detectadas por Video", "# Prendas")
    _xticks(ax_prendas)
    ax_prendas.set_ylim(0, max(pr_vals or [1]) + 2)
    # Línea de referencia en cada valor para legibilidad
    for v in set(pr_vals):
        if v > 0:
            ax_prendas.axhline(v, color="#30363D", linewidth=0.5, linestyle=":")

    # 1 ── IDs únicos por video ────────────────────────────────────────────────
    ids_vals = [s.total_ids_unicos for s in stats_list]
    bars = ax_ids.bar(x, ids_vals, bar_w, color=ACC1, alpha=0.85)
    for bar, v in zip(bars, ids_vals):
        ax_ids.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    str(v), ha="center", va="bottom", color=FG, fontsize=8)
    _style_ax(ax_ids, "IDs Únicos Raw del Tracker", "Cantidad")
    _xticks(ax_ids)

    # 2 ── FPS de procesamiento ────────────────────────────────────────────────
    fps_vals  = [s.proc_fps for s in stats_list]
    src_vals  = [s.source_fps for s in stats_list]
    w2 = bar_w / 2
    bars1 = ax_fps.bar(x - w2/2, fps_vals, w2, color=ACC2, alpha=0.85, label="Proc. FPS")
    bars2 = ax_fps.bar(x + w2/2, src_vals, w2, color=ACC5, alpha=0.85, label="FPS Fuente")
    for bar, v in zip(bars1, fps_vals):
        ax_fps.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom", color=FG, fontsize=7)
    _style_ax(ax_fps, "FPS: Procesamiento vs Fuente", "FPS")
    _xticks(ax_fps)
    ax_fps.legend(fontsize=8, facecolor="#161B22", edgecolor="#30363D", labelcolor=FG)

    # 3 ── Detecciones y tracks promedio ──────────────────────────────────────
    det_vals = [s.avg_detecciones for s in stats_list]
    trk_vals = [s.avg_tracks      for s in stats_list]
    ax_dets.bar(x - w2/2, det_vals, w2, color=ACC3, alpha=0.85, label="Avg Dets (rojo)")
    ax_dets.bar(x + w2/2, trk_vals, w2, color=ACC2, alpha=0.85, label="Avg Tracks (verde)")
    _style_ax(ax_dets, "Promedio Detecciones y Tracks por Frame", "Count / frame")
    _xticks(ax_dets)
    ax_dets.legend(fontsize=8, facecolor="#161B22", edgecolor="#30363D", labelcolor=FG)

    # 4 ── Pico máx simultáneo ────────────────────────────────────────────────
    mx_det = [s.max_dets_simultaneos   for s in stats_list]
    mx_trk = [s.max_tracks_simultaneos for s in stats_list]
    ax_tracks.bar(x - w2/2, mx_det, w2, color=ACC3, alpha=0.85, label="Max Dets")
    ax_tracks.bar(x + w2/2, mx_trk, w2, color=ACC4, alpha=0.85, label="Max Tracks")
    _style_ax(ax_tracks, "Pico Máximo de Detecciones / Tracks", "Pico")
    _xticks(ax_tracks)
    ax_tracks.legend(fontsize=8, facecolor="#161B22", edgecolor="#30363D", labelcolor=FG)

    # 5 ── Porcentaje de frames con alerta ────────────────────────────────────
    alerta_vals = [s.alertas_pct for s in stats_list]
    colors_bar  = ["#FF4444" if v > 0 else ACC6 for v in alerta_vals]
    bars3 = ax_alerta.bar(x, alerta_vals, bar_w, color=colors_bar, alpha=0.85)
    for bar, v in zip(bars3, alerta_vals):
        if v > 0:
            ax_alerta.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                           f"{v:.1f}%", ha="center", va="bottom", color=FG, fontsize=7)
    _style_ax(ax_alerta, "% Frames con ALERTA de Evasión", "% Frames")
    _xticks(ax_alerta)

    # 6 ── Distribución de estados (stacked bar) ──────────────────────────────
    all_estados = list({
        k for s in stats_list for k in s.estado_counts
    })
    # Ordenar para consistencia
    priority = ["ALERTA: EVASION", "Desalarmado OK", "Escaneando...",
                "Evaluando...", "Ocluida...", "Detectando..."]
    all_estados = sorted(
        all_estados,
        key=lambda e: priority.index(e) if e in priority else 99
    )

    bottom = np.zeros(n)
    for estado in all_estados:
        vals  = np.array([s.estado_counts.get(estado, 0) for s in stats_list], dtype=float)
        totals = np.array([max(1, sum(s.estado_counts.values())) for s in stats_list], dtype=float)
        pcts  = vals / totals * 100
        color = _ESTADO_COLORS.get(estado, "#666666")
        ax_est.bar(x, pcts, bar_w, bottom=bottom, color=color,
                   alpha=0.85, label=estado)
        bottom += pcts

    _style_ax(ax_est, "Distribución de Estados (% frames)", "% frames")
    _xticks(ax_est)
    ax_est.set_ylim(0, 105)
    ax_est.legend(fontsize=7, loc="upper right",
                  facecolor="#161B22", edgecolor="#30363D", labelcolor=FG,
                  ncol=1, bbox_to_anchor=(1.35, 1.0))

    # ── Guardar ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(img_path) or ".", exist_ok=True)
    fig.savefig(img_path, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[report] Imagen comparativa → {img_path}")
