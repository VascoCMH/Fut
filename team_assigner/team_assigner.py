"""Classificação de pessoas em equipa A, equipa B, guarda-redes e árbitros.

O detetor só precisa de encontrar pessoas: a distinção é feita aqui, pela cor
da camisola (agrupada por track) e, para quem não pertence a nenhuma das duas
cores principais, pela posição no campo.

Etapas:
1. Recortar a camisola (parte superior da bbox) em vários frames por track.
2. Remover os píxeis do relvado (verde) e calcular a cor mediana em LAB.
3. KMeans sobre as cores das tracks (com k=4, para que árbitros e
   guarda-redes não puxem os centros); as duas equipas são os dois grupos com
   mais amostras e cores distintas. Depois afinam-se os centros ignorando os
   valores afastados.
4. Cada track recebe a equipa mais votada nas suas amostras; se a maioria das
   amostras for afastada das duas equipas, fica como "outro".
5. Os "outros" que estão perto de uma baliza são guarda-redes (da equipa que
   defende esse lado); os restantes são árbitros.
"""

from collections import Counter, defaultdict

import cv2
import numpy as np

TEAM_A = 0
TEAM_B = 1
OTHER = -1

REFEREE_COLOR = (0, 255, 255)   # amarelo (BGR)
UNKNOWN_COLOR = (160, 160, 160)  # cinzento

# O L (luminosidade) pesa menos do que a cor, para tolerar sombras.
LAB_WEIGHTS = np.array([0.3, 1.0, 1.0], dtype=np.float32)


class TeamAssigner:
    def __init__(self, sample_step=5, min_samples=3, min_box_height=25,
                 outlier_factor=2.5, min_outlier_distance=20.0,
                 min_team_separation=25.0):
        self.sample_step = sample_step
        self.min_samples = min_samples
        self.min_box_height = min_box_height
        self.outlier_factor = outlier_factor
        self.min_outlier_distance = min_outlier_distance
        self.min_team_separation = min_team_separation
        self.centers = None      # centros LAB das duas equipas
        self.threshold = None    # distância máxima para pertencer a uma equipa
        self.team_colors = {}    # equipa -> cor BGR para desenhar

    # ------------------------------------------------------------------ cor
    @staticmethod
    def grass_color(frame):
        """Cor HSV típica do relvado neste frame (mediana dos 2/3 inferiores).

        Não se usa um verde fixo: há equipamentos verdes (p. ex. verde-lima)
        com o mesmo tom da relva, que só se distinguem pelo brilho/saturação.
        """
        small = cv2.resize(frame, None, fx=0.125, fy=0.125,
                           interpolation=cv2.INTER_AREA)
        lower = small[small.shape[0] // 3:]
        hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV).reshape(-1, 3)
        return np.median(hsv, axis=0)

    @staticmethod
    def shirt_color(frame, bbox, grass=None):
        """Cor mediana LAB da camisola, ou None se o recorte não servir."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        top = int(y1 + 0.15 * h)
        bottom = int(y1 + 0.50 * h)
        left = int(x1 + 0.25 * w)
        right = int(x2 - 0.25 * w)
        height, width = frame.shape[:2]
        top, bottom = max(0, top), min(height, bottom)
        left, right = max(0, left), min(width, right)
        if bottom - top < 4 or right - left < 3:
            return None
        crop = frame[top:bottom, left:right]

        if grass is None:
            grass = TeamAssigner.grass_color(frame)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int16)
        # Relvado: parecido com a relva do frame em tom, saturação e brilho.
        is_grass = (
            (np.abs(hsv[:, 0] - grass[0]) <= 10)
            & (np.abs(hsv[:, 1] - grass[1]) <= 60)
            & (np.abs(hsv[:, 2] - grass[2]) <= 50)
        )
        keep = ~is_grass
        if keep.sum() < 0.3 * len(keep):
            return None
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3)
        return np.median(lab[keep], axis=0).astype(np.float32)

    @staticmethod
    def _overlaps(bbox, others):
        x1, y1, x2, y2 = bbox
        for ox1, oy1, ox2, oy2 in others:
            if x1 < ox2 and ox1 < x2 and y1 < oy2 and oy1 < y2:
                return True
        return False

    def collect_samples(self, frames, player_tracks):
        """Recolhe cores por track ID, ignorando caixas pequenas ou sobrepostas."""
        samples = defaultdict(list)
        for frame_num in range(0, len(frames), self.sample_step):
            players = player_tracks[frame_num]
            grass = self.grass_color(frames[frame_num])
            boxes = {tid: p["bbox"] for tid, p in players.items()}
            for track_id, bbox in boxes.items():
                if bbox[3] - bbox[1] < self.min_box_height:
                    continue
                others = [b for t, b in boxes.items() if t != track_id]
                if self._overlaps(bbox, others):
                    continue
                color = self.shirt_color(frames[frame_num], bbox, grass)
                if color is not None:
                    samples[track_id].append(color)

        # Tracks curtas que não calharam nos frames amostrados: tentar todos os
        # frames, aceitando sobreposição, até juntar algumas amostras.
        for frame_num, players in enumerate(player_tracks):
            for track_id, player in players.items():
                if len(samples[track_id]) >= self.min_samples:
                    continue
                if frame_num % self.sample_step == 0 and samples[track_id]:
                    continue
                color = self.shirt_color(frames[frame_num], player["bbox"])
                if color is not None:
                    samples[track_id].append(color)
        return {t: c for t, c in samples.items() if c}

    # ------------------------------------------------------------ clustering
    def _distances(self, colors):
        diff = (colors[:, None, :] - self.centers[None, :, :]) * LAB_WEIGHTS
        return np.linalg.norm(diff, axis=2)

    def fit(self, track_colors, weights=None):
        """Encontra as cores das duas equipas.

        track_colors: cor LAB por track; weights: nº de amostras por track
        (tracks longas pesam mais do que fragmentos curtos).
        """
        data = np.array(track_colors, dtype=np.float32)
        weights = np.ones(len(data)) if weights is None else np.asarray(weights, float)
        if len(data) < 2:
            raise ValueError("Não há tracks suficientes para separar equipas.")
        scaled = data * LAB_WEIGHTS
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)

        # 1. Grupos iniciais: k=4 isola árbitros/guarda-redes dos jogadores.
        k = min(4, len(data))
        _, labels, centers = cv2.kmeans(scaled, k, None, criteria, 10,
                                        cv2.KMEANS_PP_CENTERS)
        labels = labels.ravel()
        mass = np.array([weights[labels == i].sum() for i in range(k)])
        order = np.argsort(-mass)
        first = order[0]
        second = next(
            (i for i in order[1:]
             if np.linalg.norm(centers[i] - centers[first]) > self.min_team_separation),
            order[1],
        )
        self.centers = np.array([centers[first], centers[second]]) / LAB_WEIGHTS

        # 2. Afinar: centros = mediana ponderada das tracks próximas.
        for _ in range(3):
            distances = self._distances(data)
            nearest, team = distances.min(axis=1), distances.argmin(axis=1)
            separation = np.linalg.norm(
                (self.centers[0] - self.centers[1]) * LAB_WEIGHTS)
            self.threshold = max(self.min_outlier_distance, 0.4 * separation)
            inliers = nearest <= self.threshold
            spread = np.median(nearest[inliers]) if inliers.any() else 0
            self.threshold = max(self.threshold, self.outlier_factor * spread)
            inliers = nearest <= self.threshold
            for t in (0, 1):
                members = inliers & (team == t)
                if members.any():
                    self.centers[t] = self._weighted_median(data[members],
                                                            weights[members])

        # Ordem estável: equipa A é a camisola mais clara.
        order = np.argsort(-self.centers[:, 0])
        self.centers = self.centers[order]
        for team, center in enumerate(self.centers):
            lab = np.clip(center, 0, 255).astype(np.uint8).reshape(1, 1, 3)
            bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)[0, 0]
            self.team_colors[team] = tuple(int(c) for c in bgr)
        return self

    @staticmethod
    def _weighted_median(values, weights):
        result = []
        for column in values.T:
            order = np.argsort(column)
            cumulative = np.cumsum(weights[order])
            result.append(column[order][np.searchsorted(cumulative, cumulative[-1] / 2)])
        return np.array(result, dtype=np.float32)

    def classify_samples(self, colors):
        """Equipa por amostra (OTHER se estiver longe de ambos os centros)."""
        distances = self._distances(np.array(colors, dtype=np.float32))
        labels = distances.argmin(axis=1)
        labels[distances.min(axis=1) > self.threshold] = OTHER
        return labels

    # --------------------------------------------------------------- posição
    @staticmethod
    def _median_position(player_tracks, track_id):
        positions = [
            frame[track_id]["position_m"]
            for frame in player_tracks
            if track_id in frame and frame[track_id].get("position_m") is not None
        ]
        return np.median(positions, axis=0) if positions else None

    @staticmethod
    def _x(person, use_metres):
        if use_metres:
            position = person.get("position_m")
            return None if position is None else position[0]
        x1, _, x2, _ = person["bbox"]
        return (x1 + x2) / 2

    def _goalkeeper_team(self, player_tracks, track_id, team_of):
        """Vota, frame a frame, na equipa cuja posição x média está mais perto
        do guarda-redes (cada equipa está mais perto da própria baliza).
        Usa metros quando há homografia; senão, píxeis do mesmo frame.
        """
        votes = Counter()
        for frame in player_tracks:
            if track_id not in frame:
                continue
            use_metres = frame[track_id].get("position_m") is not None
            gk_x = self._x(frame[track_id], use_metres)
            xs = {TEAM_A: [], TEAM_B: []}
            for tid, player in frame.items():
                team = team_of.get(tid)
                x = self._x(player, use_metres) if team in xs else None
                if x is not None:
                    xs[team].append(x)
            if xs[TEAM_A] and xs[TEAM_B]:
                means = {t: np.mean(v) for t, v in xs.items()}
                votes[min(means, key=lambda t: abs(means[t] - gk_x))] += 1
        return votes.most_common(1)[0][0] if votes else None

    def _resolve_others(self, player_tracks, others, team_of, yolo_hints, pitch):
        """Separa os 'outros' em guarda-redes (com equipa) e árbitros."""
        length = pitch.length / 100 if pitch is not None else 120.0
        width = pitch.width / 100 if pitch is not None else 70.0
        roles = {}
        for track_id in others:
            position = self._median_position(player_tracks, track_id)
            hint = yolo_hints.get(track_id, Counter())
            near_goal = (
                position is not None
                and (position[0] < 20 or position[0] > length - 20)
                and abs(position[1] - width / 2) < 22
            )
            hint_gk = hint["goalkeeper"] > hint["referee"]
            if not (near_goal or (position is None and hint_gk)):
                roles[track_id] = ("referee", None)
                continue
            roles[track_id] = (
                "goalkeeper", self._goalkeeper_team(player_tracks, track_id, team_of)
            )
        return roles

    # ------------------------------------------------------------- principal
    def assign(self, frames, tracks, pitch=None):
        """Acrescenta team, role e team_color a cada jogador e move os árbitros
        de tracks["players"] para tracks["referees"]. Altera tracks no lugar.

        Deve ser chamado depois da homografia (usa position_m se existir).
        """
        players = tracks["players"]
        samples = self.collect_samples(frames, players)
        reliable = {t: c for t, c in samples.items() if len(c) >= self.min_samples}
        if len(reliable) < 2:
            reliable = {t: c for t, c in samples.items() if c}
        self.fit([np.median(c, axis=0) for c in reliable.values()],
                 [len(c) for c in reliable.values()])

        yolo_hints = defaultdict(Counter)
        for frame in players:
            for tid, player in frame.items():
                yolo_hints[tid][player.get("yolo_class")] += 1

        team_of, others = {}, []
        for track_id, colors in samples.items():
            labels = self.classify_samples(colors)
            vote = Counter(labels.tolist()).most_common(1)[0][0]
            if vote == OTHER:
                others.append(track_id)
            else:
                team_of[track_id] = vote

        roles = self._resolve_others(players, others, team_of, yolo_hints, pitch)

        for frame_num, frame in enumerate(players):
            referees = tracks["referees"][frame_num]
            for track_id in list(frame):
                player = frame[track_id]
                role, team = "player", team_of.get(track_id)
                if track_id in roles:
                    role, team = roles[track_id]
                if role == "referee":
                    player.update(role="referee", team=None,
                                  team_color=REFEREE_COLOR)
                    referees[track_id] = frame.pop(track_id)
                    continue
                player["role"] = role
                player["team"] = team
                player["team_color"] = self.team_colors.get(team, UNKNOWN_COLOR)
        return tracks
