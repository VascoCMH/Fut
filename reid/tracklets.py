"""Vista por track (tracklet) construída a partir de tracks[categoria][frame][id]."""

from dataclasses import dataclass, field

import numpy as np

PLAYER_HEIGHT_M = 1.8  # para converter píxeis em metros sem homografia


@dataclass
class Tracklet:
    track_id: int
    category: str          # "players" ou "referees"
    team: object           # 0, 1 ou None
    role: str              # "player", "goalkeeper" ou "referee"
    frames: list = field(default_factory=list)
    # Extremos em metros (mediana de algumas amostras) e o frame a que se referem.
    first_pos: np.ndarray = None
    first_pos_frame: int = None
    last_pos: np.ndarray = None
    last_pos_frame: int = None
    exit_velocity: np.ndarray = None   # m/s no fim da track
    # Extremos em píxeis (para quando não há homografia).
    first_px: np.ndarray = None
    last_px: np.ndarray = None
    first_height_px: float = None
    last_height_px: float = None

    @property
    def start(self):
        return self.frames[0]

    @property
    def end(self):
        return self.frames[-1]

    @property
    def length(self):
        return len(self.frames)

    @property
    def group(self):
        """Só tracklets do mesmo grupo podem ser a mesma pessoa."""
        return (self.category, self.team, self.role)


def _edge(samples, count, from_end):
    """Mediana das primeiras/últimas `count` amostras (frame, valor)."""
    chosen = samples[-count:] if from_end else samples[:count]
    values = np.median([value for _, value in chosen], axis=0)
    frame = int(np.median([f for f, _ in chosen]))
    return values, frame


def _velocity(samples, fps, window, max_speed):
    """Velocidade (m/s) por regressão linear nas últimas `window` posições."""
    chosen = samples[-window:]
    if len(chosen) < 3:
        return np.zeros(2)
    t = np.array([f for f, _ in chosen], dtype=float) / fps
    if t[-1] - t[0] <= 0:
        return np.zeros(2)
    xy = np.array([p for _, p in chosen], dtype=float)
    velocity = np.array([np.polyfit(t, xy[:, k], 1)[0] for k in (0, 1)])
    speed = np.linalg.norm(velocity)
    if speed > max_speed:
        velocity *= max_speed / speed
    return velocity


def build_tracklets(tracks, fps, edge_samples=5, velocity_window=10,
                    max_speed=8.0, categories=("players", "referees")):
    """Devolve {track_id: Tracklet} para as categorias indicadas."""
    raw = {}
    for category in categories:
        for frame_num, frame in enumerate(tracks[category]):
            for track_id, person in frame.items():
                entry = raw.setdefault(track_id, {
                    "category": category, "team": person.get("team"),
                    "role": person.get("role", "player"),
                    "frames": [], "pos": [], "px": [], "height": [],
                })
                entry["frames"].append(frame_num)
                if person.get("position_m") is not None:
                    entry["pos"].append((frame_num, np.asarray(person["position_m"])))
                x1, y1, x2, y2 = person["bbox"]
                entry["px"].append((frame_num, np.array([(x1 + x2) / 2, y2])))
                entry["height"].append((frame_num, y2 - y1))

    tracklets = {}
    for track_id, entry in raw.items():
        t = Tracklet(int(track_id), entry["category"], entry["team"],
                     entry["role"], sorted(entry["frames"]))
        if entry["pos"]:
            t.first_pos, t.first_pos_frame = _edge(entry["pos"], edge_samples, False)
            t.last_pos, t.last_pos_frame = _edge(entry["pos"], edge_samples, True)
            t.exit_velocity = _velocity(entry["pos"], fps, velocity_window, max_speed)
        t.first_px, _ = _edge(entry["px"], edge_samples, False)
        t.last_px, _ = _edge(entry["px"], edge_samples, True)
        t.first_height_px = float(_edge(entry["height"], edge_samples, False)[0])
        t.last_height_px = float(_edge(entry["height"], edge_samples, True)[0])
        tracklets[t.track_id] = t
    return tracklets
