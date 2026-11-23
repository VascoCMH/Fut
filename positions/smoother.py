"""Suavização das trajetórias de cada pessoa (por global_id).

1. Medições suspeitas: caixas cortadas pela borda da imagem (o "centro dos
   pés" desloca-se) e saltos isolados (filtro de Hampel: mais longe da
   mediana da janela do que max(3 desvios robustos, 1 m)). São descartadas.
2. Preenchimento em linha reta entre medições válidas: tanto os pontos
   descartados como as ausências entre tracks que o ReID ligou (o jogador
   saiu de imagem e voltou).
3. Filtro Savitzky–Golay (janela de ~1 s, polinómio de grau 2) sobre a
   trajetória contínua, que dá também a velocidade (derivada).

Resultado:
- (uma track só com caixas na borda usa-as na mesma, para não perder o
  ponto de partida de uma ausência)
- em cada pessoa medida: position_smooth, speed_ms, speed_reliable e
  position_quality ("measured", "outlier" ou "border");
- speed_reliable=False na meia janela inicial e final de cada trajetória: aí
  o filtro não tem frames de um dos lados e a derivada é extrapolada (no
  vídeo de teste dava 14 m/s ao guarda-redes a entrar em imagem);
- tracks["estimated"][frame][global_id]: posições estimadas nos frames em que
  a pessoa não foi vista (ausências ligadas pelo ReID), com estimated=True.
  Servem para o minimapa, NÃO para estatísticas físicas: a linha reta
  subestima a distância e dá uma velocidade média artificial.

No vídeo de teste: velocidade bruta frame a frame com 14 % dos valores acima
de 11 m/s; depois da estabilização e desta suavização, 0 %.
"""

from collections import defaultdict

import numpy as np
from scipy.signal import savgol_filter

from .stabilizer import touches_border

CATEGORIES = ("players", "referees")


def hampel_outliers(frames, points, half=7, min_deviation=1.0, n_sigmas=3.0):
    """Máscara de pontos (N, 2) longe da mediana da janela [f-half, f+half]."""
    lo = np.searchsorted(frames, frames - half, side="left")
    hi = np.searchsorted(frames, frames + half, side="right")
    bad = np.zeros(len(frames), dtype=bool)
    for i, (a, b) in enumerate(zip(lo, hi)):
        window = points[a:b]
        if len(window) < 3:
            continue
        median = np.median(window, axis=0)
        spread = 1.4826 * np.median(np.linalg.norm(window - median, axis=1))
        if np.linalg.norm(points[i] - median) > max(n_sigmas * spread, min_deviation):
            bad[i] = True
    return bad


class PositionSmoother:
    def __init__(self, fps=25, frame_size=None, window_s=1.0, polyorder=2,
                 hampel_half=7, hampel_min_m=1.0, max_gap_s=None):
        """max_gap_s: maior ausência a preencher em linha reta (None = todas as
        que o ReID ligou)."""
        self.fps = fps or 25
        self.frame_size = frame_size
        self.window = max(5, int(round(window_s * self.fps)) | 1)  # ímpar
        self.polyorder = polyorder
        self.hampel_half = hampel_half
        self.hampel_min_m = hampel_min_m
        self.max_gap_s = max_gap_s
        self.outliers = 0
        self.border = 0
        self.estimated = 0

    def _trajectories(self, tracks):
        """{global_id: [(frame, categoria, track_id, pessoa), ...]}"""
        people = defaultdict(list)
        for category in CATEGORIES:
            for frame_num, frame in enumerate(tracks[category]):
                for track_id, person in frame.items():
                    gid = person.get("global_id", track_id)
                    people[gid].append((frame_num, category, track_id, person))
        for items in people.values():
            items.sort(key=lambda item: item[0])
        return people

    def _smooth(self, values):
        n = len(values)
        window = min(self.window, n if n % 2 else n - 1)
        if window <= self.polyorder + 1:
            return values.copy(), np.zeros_like(values)
        smooth = savgol_filter(values, window, self.polyorder, axis=0)
        velocity = savgol_filter(values, window, self.polyorder, deriv=1, axis=0) * self.fps
        return smooth, velocity

    def smooth(self, tracks):
        """Acrescenta as posições suavizadas a `tracks` (altera no lugar)."""
        n_frames = len(tracks["players"])
        tracks["estimated"] = [{} for _ in range(n_frames)]
        max_gap = None if self.max_gap_s is None else int(self.max_gap_s * self.fps)

        for gid, items in self._trajectories(tracks).items():
            frames, points, quality, track_ids = [], [], [], []
            for frame_num, _, track_id, person in items:
                if person.get("position_m") is None:
                    continue
                q = "border" if touches_border(person["bbox"], self.frame_size) else "measured"
                frames.append(frame_num)
                points.append(person["position_m"])
                quality.append(q)
                track_ids.append(track_id)
            if not frames:
                continue
            frames = np.array(frames)
            points = np.array(points, dtype=float)
            quality = np.array(quality, dtype=object)
            track_ids = np.array(track_ids)

            # Uma track (do ByteTrack) só com caixas na borda: usá-las na mesma.
            # O erro (~meia largura da caixa) é menor do que não ter nada, e
            # sem elas uma ausência ligada pelo ReID ficava sem ponto de partida.
            for track_id in np.unique(track_ids):
                mine = track_ids == track_id
                if not (quality[mine] == "measured").any():
                    quality[mine] = "measured"

            good = quality == "measured"
            if good.sum() >= 3:
                bad = hampel_outliers(frames[good], points[good], self.hampel_half,
                                      self.hampel_min_m)
                idx = np.flatnonzero(good)[bad]
                quality[idx] = "outlier"
                good[idx] = False
            self.outliers += int((quality == "outlier").sum())
            self.border += int((quality == "border").sum())
            if not good.any():
                good = np.ones(len(frames), dtype=bool)   # sem alternativa

            # Trajetória contínua: linha reta entre medições válidas.
            full = np.arange(frames[good][0], frames[good][-1] + 1)
            filled = np.column_stack([
                np.interp(full, frames[good], points[good, k]) for k in (0, 1)])
            smooth, velocity = self._smooth(filled)
            speed = np.linalg.norm(velocity, axis=1)

            seen = {}
            for (frame_num, category, track_id, person), q in zip(
                    [it for it in items if it[3].get("position_m") is not None], quality):
                seen[frame_num] = person
                person["position_quality"] = q
            edge = self.window // 2
            for k, frame_num in enumerate(full):
                if frame_num in seen:
                    person = seen[frame_num]
                    person["position_smooth"] = smooth[k].tolist()
                    person["speed_ms"] = float(speed[k])
                    person["speed_reliable"] = edge <= k < len(full) - edge
            # Ausências (frames sem a pessoa entre a 1.ª e a última medição).
            present = {it[0] for it in items}
            missing = [k for k, f in enumerate(full) if f not in present]
            if not missing:
                continue
            reference = items[0][3]
            gaps = self._gap_lengths(full, present)
            for k in missing:
                if max_gap is not None and gaps[k] > max_gap:
                    continue
                tracks["estimated"][full[k]][gid] = {
                    "global_id": gid,
                    "position_smooth": smooth[k].tolist(),
                    "speed_ms": float(speed[k]),
                    "team": reference.get("team"),
                    "role": reference.get("role"),
                    "team_color": reference.get("team_color"),
                    "estimated": True,
                    "gap_frames": int(gaps[k]),
                }
                self.estimated += 1
        return tracks

    @staticmethod
    def _gap_lengths(full, present):
        """Para cada frame de `full`, o tamanho da ausência a que pertence."""
        lengths = np.zeros(len(full), dtype=int)
        k = 0
        while k < len(full):
            if full[k] in present:
                k += 1
                continue
            start = k
            while k < len(full) and full[k] not in present:
                k += 1
            lengths[start:k] = k - start
        return lengths
