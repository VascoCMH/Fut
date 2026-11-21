"""Reidentificação: unir tracks do ByteTrack que pertencem à mesma pessoa.

Quando um jogador desaparece (oclusão, sai de imagem) e volta, o ByteTrack dá-
-lhe um ID novo. Aqui juntam-se essas tracks em cadeias A -> B -> C, cada uma com
um `global_id`, usando apenas restrições físicas:

1. Mesma equipa e mesmo papel (jogador, guarda-redes, árbitro).
2. B começa depois de A acabar (nunca coexistem), até `max_gap_s` depois.
   Guarda-redes: sem limite de tempo (só há um por equipa).
3. Distância possível: |início(B) - previsão(A)| <= raio_base + v_max * dt,
   em metros no campo (funciona com a câmara em movimento). Sem homografia,
   só para intervalos curtos e em píxeis.

As ligações são escolhidas em conjunto com o algoritmo húngaro. Ligações
ambíguas (outro candidato quase tão bom) são rejeitadas: é melhor um ID novo
do que trocar dois jogadores.
"""

from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from .tracklets import PLAYER_HEIGHT_M, build_tracklets

INVALID = np.inf


class ReIdentifier:
    def __init__(self, fps=25, max_gap_s=10.0, max_speed=8.0, base_radius_m=2.0,
                 prediction_horizon_s=1.0, pixel_max_gap_s=1.0, min_length=3,
                 ambiguity_ratio=1.25, ambiguity_margin=0.05, gap_weight=0.2,
                 goalkeeper_max_gap_s=None):
        self.fps = fps or 25
        self.max_gap_s = max_gap_s
        self.max_speed = max_speed
        self.base_radius_m = base_radius_m
        self.prediction_horizon_s = prediction_horizon_s
        self.pixel_max_gap_s = pixel_max_gap_s
        self.min_length = min_length
        self.ambiguity_ratio = ambiguity_ratio
        self.ambiguity_margin = ambiguity_margin
        self.gap_weight = gap_weight
        # Só há um guarda-redes por equipa: pode ficar muito tempo fora de
        # imagem (a câmara está do outro lado) e continuar a ser o mesmo.
        self.goalkeeper_max_gap_s = goalkeeper_max_gap_s
        self.tracklets = {}
        self.links = {}       # track_id A -> track_id B
        self.rejected = 0     # ligações válidas rejeitadas por ambiguidade

    # ---------------------------------------------------------------- custo
    def cost(self, a, b):
        """Custo de 'b continua a'; INVALID se for impossível. Entre 0 e ~1,2."""
        gap = (b.start - a.end) / self.fps
        max_gap = self.max_gap_s
        if a.role == "goalkeeper":
            max_gap = self.goalkeeper_max_gap_s or np.inf
        if gap <= 0 or gap > max_gap:
            return INVALID
        gap_penalty = self.gap_weight * min(gap / self.max_gap_s, 1.0)

        if a.last_pos is not None and b.first_pos is not None:
            dt = max((b.first_pos_frame - a.last_pos_frame) / self.fps, gap)
            horizon = min(dt, self.prediction_horizon_s)
            predicted = a.last_pos + a.exit_velocity * horizon
            distance = np.linalg.norm(b.first_pos - predicted)
            radius = self.base_radius_m + self.max_speed * dt
            return distance / radius + gap_penalty if distance <= radius else INVALID

        # Sem homografia: píxeis só valem se a câmara não teve tempo de mudar.
        if gap > self.pixel_max_gap_s:
            return INVALID
        px_per_m = np.mean([a.last_height_px, b.first_height_px]) / PLAYER_HEIGHT_M
        distance = np.linalg.norm(b.first_px - a.last_px)
        radius = (self.base_radius_m + self.max_speed * gap) * px_per_m
        return distance / radius + gap_penalty if distance <= radius else INVALID

    # ------------------------------------------------------------ ligações
    def _ambiguous(self, costs, i, j):
        c = costs[i, j]
        alternatives = np.concatenate([np.delete(costs[i], j), np.delete(costs[:, j], i)])
        alternatives = alternatives[np.isfinite(alternatives)]
        if alternatives.size == 0:
            return False
        return alternatives.min() < c * self.ambiguity_ratio + self.ambiguity_margin

    def _link_group(self, members):
        ends = sorted(members, key=lambda t: t.end)
        starts = sorted(members, key=lambda t: t.start)
        costs = np.array([[self.cost(a, b) for b in starts] for a in ends])
        if not np.isfinite(costs).any():
            return
        solver = np.where(np.isfinite(costs), costs, 1e6)
        for i, j in zip(*linear_sum_assignment(solver)):
            if not np.isfinite(costs[i, j]):
                continue
            if self._ambiguous(costs, i, j):
                self.rejected += 1
                continue
            self.links[ends[i].track_id] = starts[j].track_id

    def fit(self, tracks):
        """Calcula as ligações entre tracks (não altera `tracks`)."""
        self.tracklets = build_tracklets(tracks, self.fps, max_speed=self.max_speed)
        self.links, self.rejected = {}, 0
        groups = defaultdict(list)
        for t in self.tracklets.values():
            if t.length >= self.min_length:
                groups[t.group].append(t)
        for members in groups.values():
            if len(members) > 1:
                self._link_group(members)
        return self

    # ----------------------------------------------------------- resultado
    def global_ids(self):
        """{track_id: global_id}, numerados 1, 2, 3... por ordem de entrada."""
        predecessors = set(self.links.values())
        mapping, next_id = {}, 1
        for t in sorted(self.tracklets.values(), key=lambda t: (t.start, t.track_id)):
            if t.track_id in predecessors:
                continue
            current = t.track_id
            while current is not None:
                mapping[current] = next_id
                current = self.links.get(current)
            next_id += 1
        return mapping

    def assign(self, tracks):
        """Acrescenta `global_id` a cada pessoa em tracks. Altera no lugar."""
        self.fit(tracks)
        mapping = self.global_ids()
        for category in ("players", "referees"):
            for frame in tracks[category]:
                for track_id, person in frame.items():
                    person["global_id"] = mapping.get(int(track_id), int(track_id))
        return tracks
