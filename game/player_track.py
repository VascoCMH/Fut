"""Ficha de uma pessoa (jogador, guarda-redes ou árbitro) ao longo do vídeo.

O que é fixo (equipa, papel, cor) fica guardado uma só vez. O que muda em cada
frame fica em arrays numpy, todos com o mesmo tamanho (um elemento por frame
em que a pessoa existe, medida ou estimada).
"""

from dataclasses import dataclass, field

import numpy as np

# Valores possíveis de `quality`:
MEASURED = "measured"          # medida e usada na suavização
OUTLIER = "outlier"            # salto isolado, descartado
BORDER = "border"              # caixa cortada pela borda da imagem, descartada
ESTIMATED = "estimated"        # fora de imagem: posição em linha reta
UNCALIBRATED = "uncalibrated"  # vista, mas sem homografia nesse frame


@dataclass
class PlayerTrack:
    # Fixo
    global_id: int
    team: object = None            # 0, 1 ou None (árbitros, sem equipa)
    role: str = "player"           # "player", "goalkeeper" ou "referee"
    color: tuple = None            # cor da camisola (BGR)

    # Por frame (arrays do mesmo tamanho, ordenados por frame)
    frames: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    track_ids: np.ndarray = None   # ID do ByteTrack (-1 quando estimada)
    bbox: np.ndarray = None        # (n, 4) em píxeis; NaN quando estimada
    x_raw: np.ndarray = None       # posição medida (m); NaN sem medição
    y_raw: np.ndarray = None
    x: np.ndarray = None           # posição suavizada (m); NaN se não suavizada
    y: np.ndarray = None
    speed: np.ndarray = None       # m/s; NaN se não suavizada
    speed_reliable: np.ndarray = None
    quality: np.ndarray = None     # ver as constantes acima
    order: np.ndarray = None       # ordem dentro do frame (para desenhar igual)

    def __post_init__(self):
        n = len(self.frames)
        nan = lambda: np.full(n, np.nan)
        defaults = {
            "track_ids": lambda: np.full(n, -1, dtype=np.int64),
            "bbox": lambda: np.full((n, 4), np.nan),
            "x_raw": nan, "y_raw": nan, "x": nan, "y": nan, "speed": nan,
            "speed_reliable": lambda: np.zeros(n, dtype=bool),
            "quality": lambda: np.full(n, UNCALIBRATED, dtype=object),
            "order": lambda: np.zeros(n, dtype=np.int64),
        }
        for name, make in defaults.items():
            if getattr(self, name) is None:
                setattr(self, name, make())

    PER_FRAME = ("frames", "track_ids", "bbox", "x_raw", "y_raw", "x", "y",
                 "speed", "speed_reliable", "quality", "order")

    def __len__(self):
        return len(self.frames)

    def index_of(self, frame):
        """Posição do frame nos arrays, ou None se a pessoa não existe nesse frame."""
        k = int(np.searchsorted(self.frames, frame))
        if k < len(self.frames) and self.frames[k] == frame:
            return k
        return None

    @property
    def start(self):
        return int(self.frames[0])

    @property
    def end(self):
        return int(self.frames[-1])

    def add_frames(self, new_frames, **values):
        """Acrescenta frames (p. ex. estimados) e volta a ordenar tudo."""
        new_frames = np.asarray(new_frames, dtype=np.int64)
        n = len(new_frames)
        extra = PlayerTrack(self.global_id, frames=new_frames)
        for name, value in values.items():
            getattr(extra, name)[:] = value if np.ndim(value) else np.full(n, value)
        order = np.argsort(np.concatenate([self.frames, new_frames]), kind="stable")
        for name in self.PER_FRAME:
            joined = np.concatenate([getattr(self, name), getattr(extra, name)])
            setattr(self, name, joined[order])
