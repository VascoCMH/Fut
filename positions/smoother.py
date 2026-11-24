"""Suavização das trajetórias de cada pessoa (ficha PlayerTrack do Game).

1. Medições suspeitas: caixas cortadas pela borda da imagem (o "centro dos
   pés" desloca-se) e saltos isolados (filtro de Hampel: mais longe da
   mediana da janela do que max(3 desvios robustos, 1 m)). São descartadas.
2. Preenchimento em linha reta entre medições válidas: tanto os pontos
   descartados como as ausências entre tracks que o ReID ligou (o jogador
   saiu de imagem e voltou).
3. Filtro Savitzky–Golay (janela de ~1 s, polinómio de grau 2) sobre a
   trajetória contínua, que dá também a velocidade (derivada).

Uma track (do ByteTrack) só com caixas na borda usa-as na mesma, para não
perder o ponto de partida de uma ausência.

Resultado, nos arrays de cada PlayerTrack:
- x, y (posição suavizada), speed, speed_reliable e quality ("measured",
  "outlier", "border" ou "uncalibrated");
- speed_reliable=False na meia janela inicial e final de cada trajetória: aí
  o filtro não tem frames de um dos lados e a derivada é extrapolada (no
  vídeo de teste dava 14 m/s ao guarda-redes a entrar em imagem);
- frames novos com quality="estimated" nos frames em que a pessoa não foi
  vista (ausências ligadas pelo ReID).
  Servem para o minimapa, NÃO para estatísticas físicas: a linha reta
  subestima a distância e dá uma velocidade média artificial.

No vídeo de teste: velocidade bruta frame a frame com 14 % dos valores acima
de 11 m/s; depois da estabilização e desta suavização, 0 %.
"""

import numpy as np
from scipy.signal import savgol_filter

from game.player_track import BORDER, ESTIMATED, MEASURED, OUTLIER, UNCALIBRATED
from .stabilizer import touches_border

# Os estimados são desenhados depois de todas as pessoas medidas do frame.
ESTIMATED_ORDER = 1_000_000


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

    def _smooth(self, values):
        n = len(values)
        window = min(self.window, n if n % 2 else n - 1)
        if window <= self.polyorder + 1:
            return values.copy(), np.zeros_like(values)
        smooth = savgol_filter(values, window, self.polyorder, axis=0)
        velocity = savgol_filter(values, window, self.polyorder, deriv=1, axis=0) * self.fps
        return smooth, velocity

    def smooth(self, game):
        """Suaviza a ficha (PlayerTrack) de cada pessoa do jogo (altera no lugar)."""
        for player in game.players.values():
            self.smooth_player(player)
        return game

    def smooth_player(self, player):
        """Preenche x, y, speed, speed_reliable e quality de uma ficha e
        acrescenta os frames estimados (ausências entre medições)."""
        max_gap = None if self.max_gap_s is None else int(self.max_gap_s * self.fps)
        calibrated = np.flatnonzero(~np.isnan(player.x_raw))
        if len(calibrated) == 0:
            return player
        frames = player.frames[calibrated]
        points = np.column_stack([player.x_raw[calibrated], player.y_raw[calibrated]])
        track_ids = player.track_ids[calibrated]
        quality = np.array([BORDER if touches_border(b, self.frame_size) else MEASURED
                            for b in player.bbox[calibrated]], dtype=object)

        # Uma track (do ByteTrack) só com caixas na borda: usá-las na mesma.
        # O erro (~meia largura da caixa) é menor do que não ter nada, e
        # sem elas uma ausência ligada pelo ReID ficava sem ponto de partida.
        for track_id in np.unique(track_ids):
            mine = track_ids == track_id
            if not (quality[mine] == MEASURED).any():
                quality[mine] = MEASURED

        good = quality == MEASURED
        if good.sum() >= 3:
            bad = hampel_outliers(frames[good], points[good], self.hampel_half,
                                  self.hampel_min_m)
            idx = np.flatnonzero(good)[bad]
            quality[idx] = OUTLIER
            good[idx] = False
        self.outliers += int((quality == OUTLIER).sum())
        self.border += int((quality == BORDER).sum())
        if not good.any():
            good = np.ones(len(frames), dtype=bool)   # sem alternativa
        player.quality[calibrated] = quality

        # Trajetória contínua: linha reta entre medições válidas.
        full = np.arange(frames[good][0], frames[good][-1] + 1)
        filled = np.column_stack([
            np.interp(full, frames[good], points[good, k]) for k in (0, 1)])
        smooth, velocity = self._smooth(filled)
        speed = np.linalg.norm(velocity, axis=1)
        edge = self.window // 2
        reliable = (np.arange(len(full)) >= edge) & (np.arange(len(full)) < len(full) - edge)

        # Frames medidos dentro do intervalo suavizado.
        inside = np.flatnonzero((player.frames >= full[0]) & (player.frames <= full[-1])
                                & (player.quality != UNCALIBRATED))
        k = player.frames[inside] - full[0]
        player.x[inside], player.y[inside] = smooth[k, 0], smooth[k, 1]
        player.speed[inside] = speed[k]
        player.speed_reliable[inside] = reliable[k]

        # Ausências (frames sem a pessoa entre a 1.ª e a última medição).
        present = set(player.frames.tolist())
        missing = np.array([i for i, f in enumerate(full) if f not in present], dtype=int)
        if len(missing) == 0:
            return player
        gaps = self._gap_lengths(full, present)
        if max_gap is not None:
            missing = missing[gaps[missing] <= max_gap]
        if len(missing) == 0:
            return player
        player.add_frames(
            full[missing], x=smooth[missing, 0], y=smooth[missing, 1],
            speed=speed[missing], quality=ESTIMATED,
            order=ESTIMATED_ORDER + player.global_id)
        self.estimated += len(missing)
        return player

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
