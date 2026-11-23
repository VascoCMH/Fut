"""Estabilização da homografia frame a frame, usando os próprios jogadores.

As posições estão em metros no campo, por isso o movimento da câmara não as
devia mudar. Quando todos os jogadores de um frame "saltam" juntos, o erro é
da homografia desse frame (keypoints mal detetados), não movimento real.

Para cada frame:
1. cada pessoa tem uma posição de referência robusta: a mediana da sua
   trajetória numa janela de tempo à volta do frame;
2. estima-se, com RANSAC, a transformação de semelhança (translação, rotação
   e escala) que leva as posições do frame para as referências; como é comum
   a todos, é o erro da homografia;
3. essa correção é aplicada à própria matriz (correção @ matriz), para que
   jogadores, bola e tudo o resto fiquem coerentes.
Repete-se com janelas cada vez mais pequenas.

No vídeo de teste, o deslocamento comum a todos os jogadores no percentil 99
passou de 79 m/s para 21 m/s.
"""

from collections import defaultdict

import cv2
import numpy as np


def _feet(bbox):
    x1, _, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, y2], dtype=float)


def _project(matrix, points):
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def touches_border(bbox, frame_size, margin=3):
    """Caixa cortada pela borda da imagem: o 'centro dos pés' não é fiável."""
    if frame_size is None:
        return False
    width, height = frame_size
    x1, _, x2, y2 = bbox
    return x1 <= margin or x2 >= width - margin or y2 >= height - margin


def _rolling_median(frames, points, half):
    """Mediana de `points` em [f - half, f + half] para cada frame (O(n·w))."""
    lo = np.searchsorted(frames, frames - half, side="left")
    hi = np.searchsorted(frames, frames + half, side="right")
    out = np.empty_like(points)
    valid = np.zeros(len(frames), dtype=bool)
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b - a >= 3:
            out[i] = np.median(points[a:b], axis=0)
            valid[i] = True
    return out, valid


class HomographyStabilizer:
    def __init__(self, windows=(12, 6), min_people=5, ransac_threshold_m=1.0,
                 frame_size=None):
        """windows: meias-janelas (frames) da referência, uma por iteração."""
        self.windows = windows
        self.min_people = min_people
        self.ransac_threshold_m = ransac_threshold_m
        self.frame_size = frame_size
        self.corrections = []   # deslocamento médio aplicado por frame (m)

    def _observations(self, tracks):
        """{track_id: (frames, feet_px)} de todas as pessoas."""
        obs = defaultdict(lambda: ([], []))
        for category in ("players", "referees"):
            for frame_num, frame in enumerate(tracks[category]):
                for track_id, person in frame.items():
                    if touches_border(person["bbox"], self.frame_size):
                        continue
                    obs[track_id][0].append(frame_num)
                    obs[track_id][1].append(_feet(person["bbox"]))
        return {t: (np.array(f), np.array(p)) for t, (f, p) in obs.items() if len(f) >= 3}

    def stabilize(self, tracks, matrices):
        """Devolve uma nova lista de matrizes corrigidas (não altera a original)."""
        matrices = [None if m is None else np.array(m, dtype=float) for m in matrices]
        observations = self._observations(tracks)
        n = len(matrices)
        self.corrections = [0.0] * n

        for half in self.windows:
            # Posições atuais (com as matrizes já corrigidas) e referências.
            per_frame = defaultdict(lambda: ([], []))
            for frames, feet in observations.values():
                keep = np.array([matrices[f] is not None for f in frames])
                if keep.sum() < 3:
                    continue
                fr, ft = frames[keep], feet[keep]
                positions = np.array([_project(matrices[f], p[None])[0] for f, p in zip(fr, ft)])
                reference, valid = _rolling_median(fr, positions, half)
                for f, pos, ref, ok in zip(fr, positions, reference, valid):
                    if ok:
                        per_frame[f][0].append(pos)
                        per_frame[f][1].append(ref)

            for f, (src, dst) in per_frame.items():
                if len(src) < self.min_people:
                    continue
                src, dst = np.float32(src), np.float32(dst)
                affine, _ = cv2.estimateAffinePartial2D(
                    src, dst, method=cv2.RANSAC,
                    ransacReprojThreshold=self.ransac_threshold_m)
                if affine is None or not np.isfinite(affine).all():
                    continue
                correction = np.vstack([affine, [0, 0, 1]])
                matrices[f] = correction @ matrices[f]
                moved = _project(correction, src.astype(float)) - src
                self.corrections[f] += float(np.mean(np.linalg.norm(moved, axis=1)))
        return matrices
