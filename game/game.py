"""O jogo: as fichas de todas as pessoas e os dados do jogo por frame.

- `players`: {global_id: PlayerTrack}, com os dados de movimento.
- `possession`: equipa com a posse em cada frame (0, 1, "disputed" ou None).
- `ball_holder`: global_id de quem tem a bola em cada frame (ou None).
- `positions_at(frame)`: todas as pessoas nesse frame (para a posse e o desenho).
- `to_csv(path)`: uma linha por pessoa e por frame.
"""

import csv
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .player_track import ESTIMATED, MEASURED, UNCALIBRATED, PlayerTrack


@dataclass
class PersonState:
    """Uma pessoa num frame (vista montada a partir da ficha)."""
    global_id: int
    team: object
    role: str
    color: tuple
    track_id: object         # ID do ByteTrack, ou None se estimada
    bbox: object             # [x1, y1, x2, y2] ou None se estimada
    position_raw: object     # (x, y) medida em metros, ou None
    position: object         # (x, y) suavizada em metros, ou None
    speed: object            # m/s, ou None
    speed_reliable: bool
    quality: str
    has_ball: bool

    @property
    def estimated(self):
        return self.quality == ESTIMATED


def _pair(x, y):
    return None if np.isnan(x) or np.isnan(y) else (float(x), float(y))


class Game:
    def __init__(self, fps, n_frames, players=None):
        self.fps = fps
        self.n_frames = n_frames
        self.players = players or {}
        self.possession = [None] * n_frames
        self.ball_holder = [None] * n_frames

    @classmethod
    def from_tracks(cls, tracks, fps):
        """Converte o formato por frame (tracks["players"/"referees"]) em fichas.

        Corre depois do ReID: agrupa pelo global_id.
        """
        n_frames = len(tracks["players"])
        rows = defaultdict(list)
        info = {}
        for rank, category in enumerate(("players", "referees")):
            for frame_num, frame in enumerate(tracks[category]):
                for order, (track_id, person) in enumerate(frame.items()):
                    gid = int(person.get("global_id", track_id))
                    rows[gid].append((frame_num, int(track_id), person, rank * 10_000 + order))
                    info.setdefault(gid, person)
        players = {}
        for gid, items in rows.items():
            items.sort(key=lambda item: item[0])
            first = info[gid]
            raw = np.array([p["position_m"] if p.get("position_m") is not None
                            else (np.nan, np.nan) for _, _, p, _ in items], dtype=float)
            track = PlayerTrack(
                global_id=gid, team=first.get("team"),
                role=first.get("role", "player"), color=first.get("team_color"),
                frames=np.array([f for f, _, _, _ in items], dtype=np.int64),
                track_ids=np.array([t for _, t, _, _ in items], dtype=np.int64),
                bbox=np.array([p["bbox"] for _, _, p, _ in items], dtype=float),
                x_raw=raw[:, 0], y_raw=raw[:, 1],
                order=np.array([o for _, _, _, o in items], dtype=np.int64),
            )
            track.quality[~np.isnan(track.x_raw)] = MEASURED
            players[gid] = track
        return cls(fps, n_frames, players)

    # ---------------------------------------------------------------- vistas
    def positions_at(self, frame):
        """Pessoas presentes no frame (medidas e estimadas), pela ordem em que
        foram detetadas (jogadores primeiro, árbitros depois)."""
        holder = self.ball_holder[frame] if 0 <= frame < self.n_frames else None
        people = []
        for gid, p in self.players.items():
            k = p.index_of(frame)
            if k is None:
                continue
            estimated = p.quality[k] == ESTIMATED
            people.append((p.order[k], PersonState(
                global_id=gid, team=p.team, role=p.role, color=p.color,
                track_id=None if estimated else int(p.track_ids[k]),
                bbox=None if estimated else p.bbox[k].tolist(),
                position_raw=_pair(p.x_raw[k], p.y_raw[k]),
                position=_pair(p.x[k], p.y[k]),
                speed=None if np.isnan(p.speed[k]) else float(p.speed[k]),
                speed_reliable=bool(p.speed_reliable[k]),
                quality=str(p.quality[k]),
                has_ball=holder == gid,
            )))
        people.sort(key=lambda item: item[0])
        return [person for _, person in people]

    # ------------------------------------------------------------ exportar
    def to_csv(self, path):
        """Uma linha por pessoa e por frame, ordenadas por frame."""
        columns = ["frame", "time_s", "global_id", "team", "role", "quality",
                   "x", "y", "x_raw", "y_raw", "speed_ms", "speed_reliable",
                   "track_id", "has_ball", "possession"]

        def fmt(value):
            return "" if value is None or (isinstance(value, float) and np.isnan(value)) \
                else round(float(value), 3)

        rows = []
        for gid, p in self.players.items():
            for k, frame in enumerate(p.frames):
                frame = int(frame)
                rows.append([
                    frame, round(frame / self.fps, 3) if self.fps else "", gid,
                    "" if p.team is None else p.team, p.role, p.quality[k],
                    fmt(p.x[k]), fmt(p.y[k]), fmt(p.x_raw[k]), fmt(p.y_raw[k]),
                    fmt(p.speed[k]), bool(p.speed_reliable[k]),
                    "" if p.track_ids[k] < 0 else int(p.track_ids[k]),
                    self.ball_holder[frame] == gid,
                    "" if self.possession[frame] is None else self.possession[frame],
                ])
        rows.sort(key=lambda row: (row[0], row[2]))
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
        return path
