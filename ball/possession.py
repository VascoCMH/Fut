"""Posse de bola por frame: equipa 0, equipa 1 ou disputada.

Por frame (estado "bruto"), com a distância dos pés de cada jogador à bola:
- ninguém a menos de `control_radius_m`  -> "livre" (bola em passe/remate);
- jogadores das duas equipas dentro do raio e com distâncias parecidas
  (diferença < `dispute_margin_m`)       -> "disputada";
- caso contrário                          -> equipa do jogador mais próximo
                                             (game.ball_holder = global_id dele).

Posse final (suavizada):
- "livre" mantém a posse anterior: num passe, a bola continua da equipa que
  a jogou até alguém a receber;
- uma mudança só é aceite se o novo estado durar `min_frames` seguidos, para
  evitar oscilações quando os jogadores se cruzam;
- posições interpoladas em lacunas longas (> `max_trusted_gap` frames) não
  mudam a posse: a bola falha sobretudo quando está nos pés de alguém, e a
  linha reta entre deteções pode passar longe de quem a tem;
- sem bola (lacuna maior do que a interpolação) a posse fica None.

Distâncias em metros quando há homografia; senão em píxeis convertidos para
metros pela altura do jogador (~1,8 m).
"""

from collections import Counter

import numpy as np

PLAYER_HEIGHT_M = 1.8
DISPUTED = "disputed"
FREE = "free"


class PossessionAssigner:
    def __init__(self, control_radius_m=2.5, dispute_margin_m=1.0, min_frames=5,
                 max_trusted_gap=8):
        self.control_radius_m = control_radius_m
        self.max_trusted_gap = max_trusted_gap
        self.dispute_margin_m = dispute_margin_m
        self.min_frames = min_frames
        self.raw = []          # estado bruto por frame
        self.possession = []   # posse suavizada por frame
        self.possessor = []    # global_id do jogador com a bola (ou None)

    @staticmethod
    def _distance(ball, player):
        """Distância (m) entre a bola e os pés do jogador (PersonState)."""
        if ball.get("position_m") is not None and player.position_raw is not None:
            return float(np.linalg.norm(
                np.asarray(ball["position_m"]) - np.asarray(player.position_raw)))
        x1, y1, x2, y2 = player.bbox
        feet = np.array([(x1 + x2) / 2, y2])
        bx1, _, bx2, by2 = ball["bbox"]
        ball_px = np.array([(bx1 + bx2) / 2, by2])
        px_per_m = max(y2 - y1, 1.0) / PLAYER_HEIGHT_M
        return float(np.linalg.norm(feet - ball_px) / px_per_m)

    def frame_state(self, ball, players):
        """(estado bruto, global_id do mais próximo) de um frame.

        players: lista de PersonState medidas nesse frame."""
        if ball is None:
            return None, None
        if ball.get("interpolated") and ball.get("gap_frames", 0) > self.max_trusted_gap:
            return FREE, None
        nearest = {}  # equipa -> (distância, global_id)
        for player in players:
            team = player.team
            if team not in (0, 1):
                continue
            d = self._distance(ball, player)
            if team not in nearest or d < nearest[team][0]:
                nearest[team] = (d, player.global_id)
        close = {t: v for t, v in nearest.items() if v[0] <= self.control_radius_m}
        if not close:
            return FREE, None
        if len(close) == 2 and abs(close[0][0] - close[1][0]) < self.dispute_margin_m:
            return DISPUTED, None
        team = min(close, key=lambda t: close[t][0])
        return team, close[team][1]

    def assign(self, game, ball_tracks):
        """Calcula a posse de todos os frames e guarda-a no jogo
        (game.possession e game.ball_holder, por frame). Acrescenta também
        `possession` à bola. Devolve a lista de posse."""
        self.raw, self.possession, self.possessor = [], [], []
        current, candidate, streak = None, None, 0
        for frame_num in range(game.n_frames):
            players = [p for p in game.positions_at(frame_num)
                       if not p.estimated and p.role != "referee"]
            ball = ball_tracks[frame_num].get(1)
            state, holder = self.frame_state(ball, players)
            self.raw.append(state)

            if state is None:
                current, candidate, streak = None, None, 0
            elif state != FREE:
                if current is None:
                    current = state            # primeira observação: aceitar já
                elif state == current:
                    candidate, streak = None, 0
                else:
                    streak = streak + 1 if state == candidate else 1
                    candidate = state
                    if streak >= self.min_frames:
                        current, candidate, streak = state, None, 0
            self.possession.append(current)

            holder = holder if current == state and current in (0, 1) else None
            self.possessor.append(holder)
            if ball is not None:
                ball["possession"] = current
        game.possession = list(self.possession)
        game.ball_holder = list(self.possessor)
        return self.possession

    def statistics(self):
        """Percentagem de frames (com posse conhecida) de cada estado."""
        known = [p for p in self.possession if p is not None]
        counts = Counter(known)
        total = len(known)
        return {
            state: (100.0 * counts.get(state, 0) / total if total else 0.0)
            for state in (0, 1, DISPUTED)
        }
