"""
modules/consolidador.py
=======================
Post-procesa todos los IDs generados por el tracker en un video para:

  1. Filtrar IDs efímeros (ruido, < TRACK_MIN_STABLE_FRAMES)
  2. Fusionar IDs que el tracker fragmentó y que corresponden a la MISMA prenda
     (criterio: no solapamiento temporal + alta similitud de color)
  3. Devolver el conteo final de PRENDAS REALES en escena

Esto es independiente del tracker usado y actúa como capa de corrección.
"""

import cv2
import numpy as np
import config


class IdConsolidador:
    """
    Se instancia por video. Se llama a `registrar()` frame a frame y
    a `consolidar()` al finalizar el video.
    """

    def __init__(self):
        # id_est → dict con métricas acumuladas
        self._datos = {}

    # ──────────────────────────────────────────────────────────────────────────
    # REGISTRO FRAME A FRAME
    # ──────────────────────────────────────────────────────────────────────────

    def registrar(self, id_est: int, frame_count: int,
                  histograma, cx: int, cy: int, confirmado: bool):
        """Registra la presencia de un ID en un frame dado."""
        if id_est not in self._datos:
            self._datos[id_est] = {
                'first_frame':  frame_count,
                'last_frame':   frame_count,
                'frames_vistos': 0,
                'confirmado':   False,
                'histogramas':  [],      # muestra representativa de histogramas
                'cx_sum':       0,
                'cy_sum':       0,
            }
        d = self._datos[id_est]
        d['last_frame']    = frame_count
        d['frames_vistos'] += 1
        d['confirmado']    = d['confirmado'] or confirmado
        d['cx_sum']       += cx
        d['cy_sum']       += cy

        # Guardar histograma cada N frames para no saturar memoria
        if histograma is not None and d['frames_vistos'] % 5 == 0:
            d['histogramas'].append(histograma.copy())

    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS INTERNOS
    # ──────────────────────────────────────────────────────────────────────────

    def _sim_apariencia(self, hists_a, hists_b) -> float:
        """Correlación media entre conjuntos de histogramas."""
        if not hists_a or not hists_b:
            return 0.5   # neutral sin datos
        # Comparar el histograma central de cada ID
        ha = hists_a[len(hists_a) // 2]
        hb = hists_b[len(hists_b) // 2]
        return float(max(0.0, cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)))

    def _pos_media(self, d) -> tuple:
        n = max(1, d['frames_vistos'])
        return (d['cx_sum'] / n, d['cy_sum'] / n)

    # ──────────────────────────────────────────────────────────────────────────
    # UNION-FIND
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_uf(ids):
        parent = {i: i for i in ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        return find, union

    # ──────────────────────────────────────────────────────────────────────────
    # CONSOLIDACIÓN PRINCIPAL
    # ──────────────────────────────────────────────────────────────────────────

    def consolidar(self):
        """
        Devuelve:
          grupos          dict { root_id: [id1, id2, ...] }
          prendas_reales  int  — número de prendas únicas en escena
          ids_validos     set  — IDs que pasaron el filtro mínimo
        """
        min_frames   = config.TRACK_MIN_STABLE_FRAMES
        max_gap      = config.CONSOLIDATION_MAX_GAP
        sim_thresh   = config.CONSOLIDATION_SIM_THRESH

        # ── 1. Filtrar IDs ruidosos ───────────────────────────────────────────
        ids_validos_dict = {
            ide: d for ide, d in self._datos.items()
            if d['frames_vistos'] >= min_frames and d['confirmado']
        }

        if not ids_validos_dict:
            return {}, 0, set()

        ids_list = list(ids_validos_dict.keys())
        n = len(ids_list)

        find, union = self._make_uf(ids_list)

        # ── 2. Buscar pares candidatos a fusión ───────────────────────────────
        for i in range(n):
            for j in range(i + 1, n):
                id_a = ids_list[i]
                id_b = ids_list[j]
                da   = ids_validos_dict[id_a]
                db   = ids_validos_dict[id_b]

                # ¿Estuvieron simultáneamente activos?
                overlap = (da['first_frame'] <= db['last_frame'] and
                           db['first_frame'] <= da['last_frame'])
                if overlap:
                    # Observados al mismo tiempo → definitivamente distintos
                    continue

                # Brecha temporal entre los dos IDs
                gap = max(da['first_frame'], db['first_frame']) - \
                      min(da['last_frame'],  db['last_frame'])
                if gap > max_gap:
                    continue   # demasiado separados en el tiempo

                # Similitud de apariencia
                sim = self._sim_apariencia(da['histogramas'], db['histogramas'])
                if sim >= sim_thresh:
                    union(id_a, id_b)

        # ── 3. Agrupar por componente conexa ──────────────────────────────────
        grupos: dict = {}
        for ide in ids_list:
            root = find(ide)
            grupos.setdefault(root, []).append(ide)

        prendas_reales = len(grupos)
        ids_validos    = set(ids_validos_dict.keys())

        return grupos, prendas_reales, ids_validos

    # ──────────────────────────────────────────────────────────────────────────
    # ESTADÍSTICAS POR PRENDA
    # ──────────────────────────────────────────────────────────────────────────

    def resumen_prendas(self, grupos: dict, ids_validos_dict: dict = None) -> list:
        """
        Devuelve lista de dicts con info de cada prenda consolidada:
          { 'prenda': N, 'ids': [...], 'first_frame': F, 'last_frame': L,
            'frames_visible': F, 'cx': cx, 'cy': cy }
        """
        if ids_validos_dict is None:
            ids_validos_dict = self._datos

        result = []
        for prenda_n, (root, miembros) in enumerate(grupos.items(), start=1):
            first_f = min(ids_validos_dict[m]['first_frame'] for m in miembros)
            last_f  = max(ids_validos_dict[m]['last_frame']  for m in miembros)
            frames  = sum(ids_validos_dict[m]['frames_vistos'] for m in miembros)
            cx = np.mean([self._pos_media(ids_validos_dict[m])[0] for m in miembros])
            cy = np.mean([self._pos_media(ids_validos_dict[m])[1] for m in miembros])
            result.append({
                'prenda':       prenda_n,
                'ids':          sorted(miembros),
                'first_frame':  first_f,
                'last_frame':   last_f,
                'frames_visible': frames,
                'cx':           int(cx),
                'cy':           int(cy),
            })

        result.sort(key=lambda x: x['first_frame'])
        for i, r in enumerate(result, start=1):
            r['prenda'] = i

        return result
