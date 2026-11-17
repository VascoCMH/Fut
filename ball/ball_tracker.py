"""Limpeza e interpolação da trajetória da bola.

1. Limpeza: o detetor dá falsos positivos (saltos impossíveis de centenas de
   píxeis num frame). A trajetória é dividida em segmentos coerentes; segmentos
   curtos incoerentes com os vizinhos dos dois lados são removidos. Nos frames
   com várias candidatas, escolhe-se a que é coerente com os vizinhos.
2. Interpolação: nas lacunas até `max_gap_frames`, assume-se que a bola se
   moveu a velocidade constante em linha reta. Se houver homografia em toda a
   lacuna, interpola-se em metros no campo e projeta-se para a imagem de cada
   frame (correto mesmo com a câmara em movimento); senão, em píxeis.
   As posições criadas ficam marcadas com interpolated=True e gap_frames.
   Atenção: a bola falha sobretudo quando está nos pés de um jogador (tapada),
   e aí o movimento real não é retilíneo; nas lacunas longas a posição
   interpolada serve para desenhar, não para decidir a posse.
"""

import numpy as np


def _feet(bbox):
    x1, _, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, y2], dtype=float)


class BallTracker:
    def __init__(self, max_speed_px=70.0, base_jump_px=30.0, max_gap_frames=50,
                 max_outlier_length=3):
        """max_speed_px: deslocação máxima por frame (píxeis); base_jump_px:
        tolerância fixa; max_gap_frames: maior lacuna a preencher (50 = 2 s a
        25 fps); max_outlier_length: segmentos até este tamanho podem ser
        removidos se forem incoerentes com os dois vizinhos.
        """
        self.max_speed_px = max_speed_px
        self.base_jump_px = base_jump_px
        self.max_gap_frames = max_gap_frames
        self.max_outlier_length = max_outlier_length
        self.removed = 0
        self.recovered = 0
        self.interpolated = 0

    # ------------------------------------------------------------- limpeza
    def _consistent(self, frame_a, point_a, frame_b, point_b):
        gap = abs(frame_b - frame_a)
        limit = self.base_jump_px + self.max_speed_px * gap
        return np.linalg.norm(point_b - point_a) <= limit

    def _segments(self, frames, points):
        segments, current = [], [frames[0]]
        for prev, cur in zip(frames, frames[1:]):
            if self._consistent(prev, points[prev], cur, points[cur]):
                current.append(cur)
            else:
                segments.append(current)
                current = [cur]
        segments.append(current)
        return segments

    def clean(self, ball_tracks):
        """Remove falsos positivos de tracks["ball"] (altera no lugar)."""
        candidates_by_frame = {
            f: b[1]["candidates"] for f, b in enumerate(ball_tracks)
            if 1 in b and len(b[1].get("candidates", [])) > 1
        }
        points = {f: _feet(b[1]["bbox"]) for f, b in enumerate(ball_tracks) if 1 in b}
        if len(points) < 2:
            return ball_tracks
        changed = True
        while changed:
            changed = False
            frames = sorted(points)
            segments = self._segments(frames, points)
            for k, segment in enumerate(segments):
                if len(segment) > self.max_outlier_length or len(segments) == 1:
                    continue
                prev = segments[k - 1][-1] if k > 0 else None
                nxt = segments[k + 1][0] if k + 1 < len(segments) else None
                # Coerente com algum vizinho? Então é a trajetória que mudou.
                ok_prev = prev is not None and self._consistent(
                    prev, points[prev], segment[0], points[segment[0]])
                ok_next = nxt is not None and self._consistent(
                    segment[-1], points[segment[-1]], nxt, points[nxt])
                if ok_prev or ok_next:
                    continue
                # Só remover se os vizinhos forem mais longos (mais fiáveis).
                neighbours = [s for s in (segments[k - 1] if k > 0 else None,
                                          segments[k + 1] if k + 1 < len(segments) else None)
                              if s is not None]
                if all(len(s) <= len(segment) for s in neighbours):
                    continue
                for f in segment:
                    del points[f]
                    ball_tracks[f].pop(1)
                    self.removed += 1
                changed = True
                break

        self._recover_candidates(ball_tracks, points, candidates_by_frame)
        return ball_tracks

    def _recover_candidates(self, ball_tracks, points, candidates_by_frame):
        """Frames sem bola (após a limpeza) mas com outras candidatas: aceitar
        a mais confiante que seja coerente com os vizinhos válidos."""
        frames = sorted(points)
        if not frames:
            return
        for f, candidates in sorted(candidates_by_frame.items()):
            if 1 in ball_tracks[f]:
                continue
            i = np.searchsorted(frames, f)
            prev = frames[i - 1] if i > 0 else None
            nxt = frames[i] if i < len(frames) else None
            for candidate in sorted(candidates, key=lambda c: -c["confidence"]):
                p = _feet(candidate["bbox"])
                if ((prev is None or self._consistent(prev, points[prev], f, p))
                        and (nxt is None or self._consistent(f, p, nxt, points[nxt]))
                        and (prev is not None or nxt is not None)):
                    ball_tracks[f][1] = {**candidate, "candidates": candidates}
                    points[f] = p
                    self.recovered += 1
                    break

    # -------------------------------------------------------- interpolação
    def interpolate(self, ball_tracks, matrices=None):
        """Preenche lacunas curtas por movimento retilíneo uniforme.

        matrices: lista (por frame) de homografias píxeis -> metros, ou None.
        """
        frames = [f for f, b in enumerate(ball_tracks) if 1 in b]
        for a, b in zip(frames, frames[1:]):
            gap = b - a - 1
            if gap == 0 or gap > self.max_gap_frames:
                continue
            ball_a, ball_b = ball_tracks[a][1], ball_tracks[b][1]
            box_a, box_b = np.asarray(ball_a["bbox"]), np.asarray(ball_b["bbox"])
            size_a, size_b = box_a[2:] - box_a[:2], box_b[2:] - box_b[:2]
            in_metres = (
                matrices is not None
                and ball_a.get("position_m") is not None
                and ball_b.get("position_m") is not None
                and all(matrices[f] is not None for f in range(a, b + 1))
            )
            for f in range(a + 1, b):
                t = (f - a) / (b - a)
                size = (1 - t) * size_a + t * size_b
                entry = {"interpolated": True, "gap_frames": gap}
                if in_metres:
                    pos = (1 - t) * np.asarray(ball_a["position_m"]) + \
                        t * np.asarray(ball_b["position_m"])
                    projected = np.linalg.inv(matrices[f]) @ np.array([*pos, 1.0])
                    feet = projected[:2] / projected[2]
                    entry["position_m"] = pos.tolist()
                else:
                    feet = (1 - t) * _feet(box_a) + t * _feet(box_b)
                    entry["position_m"] = None
                entry["feet_px"] = feet.tolist()
                entry["bbox"] = [feet[0] - size[0] / 2, feet[1] - size[1],
                                 feet[0] + size[0] / 2, feet[1]]
                ball_tracks[f][1] = entry
                self.interpolated += 1
        return ball_tracks

    def process(self, ball_tracks, matrices=None):
        """Limpeza seguida de interpolação."""
        self.clean(ball_tracks)
        self.interpolate(ball_tracks, matrices)
        return ball_tracks
