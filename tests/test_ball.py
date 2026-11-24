"""Testes sintéticos da limpeza, interpolação e posse de bola."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ball import DISPUTED, BallTracker, PossessionAssigner  # noqa: E402
from game import Game  # noqa: E402


def ball_box(x, y, size=14):
    return [x - size / 2, y - size, x + size / 2, y]


def ball_tracks(points, n_frames):
    """points: {frame: (x, y)} em píxeis."""
    tracks = [{} for _ in range(n_frames)]
    for f, (x, y) in points.items():
        box = ball_box(x, y)
        tracks[f][1] = {"bbox": box, "confidence": 0.8,
                        "candidates": [{"bbox": box, "confidence": 0.8}]}
    return tracks


def feet(entry):
    x1, _, x2, y2 = entry["bbox"]
    return np.array([(x1 + x2) / 2, y2])


# ------------------------------------------------------------------ limpeza
def test_removes_isolated_false_positive():
    points = {f: (100 + 10 * f, 500) for f in range(20)}
    points[10] = (1500, 100)  # salto impossível
    tracks = ball_tracks(points, 20)
    tracker = BallTracker()
    tracker.clean(tracks)
    assert 1 not in tracks[10]
    assert tracker.removed == 1
    assert all(1 in tracks[f] for f in range(20) if f != 10)


def test_removes_short_cluster_of_false_positives():
    points = {f: (100 + 10 * f, 500) for f in range(30)}
    for f in (14, 15, 16):
        points[f] = (1600 + f, 120)  # três deteções seguidas noutro sítio
    tracks = ball_tracks(points, 30)
    BallTracker().clean(tracks)
    assert all(1 not in tracks[f] for f in (14, 15, 16))
    assert 1 in tracks[13] and 1 in tracks[17]


def test_keeps_real_fast_movement():
    # Remate: 60 px por frame é rápido mas possível.
    points = {f: (100 + 60 * f, 500) for f in range(20)}
    tracks = ball_tracks(points, 20)
    tracker = BallTracker()
    tracker.clean(tracks)
    assert tracker.removed == 0


def test_recovers_other_candidate_when_best_is_wrong():
    points = {f: (100 + 10 * f, 500) for f in range(20)}
    tracks = ball_tracks(points, 20)
    good = {"bbox": ball_box(200, 500), "confidence": 0.4}
    wrong = {"bbox": ball_box(1500, 100), "confidence": 0.9}
    tracks[10][1] = {**wrong, "candidates": [wrong, good]}
    tracker = BallTracker()
    tracker.clean(tracks)
    assert np.allclose(feet(tracks[10][1]), (200, 500))
    assert tracker.recovered == 1


# ------------------------------------------------------------- interpolação
def test_interpolates_gap_in_straight_line_pixels():
    tracks = ball_tracks({0: (100, 500), 10: (300, 400)}, 11)
    BallTracker().interpolate(tracks)
    for f in range(1, 10):
        assert tracks[f][1]["interpolated"]
        assert np.allclose(feet(tracks[f][1]), (100 + 20 * f, 500 - 10 * f))


def test_does_not_fill_long_gaps_or_edges():
    tracks = ball_tracks({5: (100, 500), 100: (300, 400)}, 110)
    BallTracker(max_gap_frames=50).interpolate(tracks)
    assert all(1 not in tracks[f] for f in range(0, 5))      # antes da 1.ª
    assert all(1 not in tracks[f] for f in range(6, 100))     # lacuna longa
    assert all(1 not in tracks[f] for f in range(101, 110))   # depois da última


def test_interpolates_in_metres_with_moving_camera():
    # Bola parada no campo em (50, 30) m enquanto a câmara se desloca:
    # em píxeis "mexe-se", mas a posição interpolada em metros fica parada
    # e a projeção acompanha a câmara.
    n = 11
    matrices = []
    for f in range(n):
        shift = 20 * f  # a câmara anda 20 px por frame
        to_px = np.array([[10, 0, 100 - shift], [0, 10, 100], [0, 0, 1]], float)
        matrices.append(np.linalg.inv(to_px))
    tracks = [{} for _ in range(n)]
    for f in (0, 10):
        px = np.linalg.inv(matrices[f]) @ np.array([50, 30, 1.0])
        tracks[f][1] = {"bbox": ball_box(px[0], px[1]), "position_m": [50.0, 30.0]}
    BallTracker().interpolate(tracks, matrices)
    for f in range(1, 10):
        assert np.allclose(tracks[f][1]["position_m"], (50, 30))
        assert np.allclose(feet(tracks[f][1]), (600 - 20 * f, 400))


# -------------------------------------------------------------------- posse
def tracks_with(ball_positions, players):
    """ball_positions: lista de (x, y) metros ou None; players: {id: (team, [pos por frame])}."""
    n = len(ball_positions)
    tracks = {"players": [{} for _ in range(n)], "referees": [{} for _ in range(n)],
              "ball": [{} for _ in range(n)]}
    for f, pos in enumerate(ball_positions):
        if pos is not None:
            tracks["ball"][f][1] = {"bbox": [0, 0, 1, 1], "position_m": list(pos)}
    for track_id, (team, positions) in players.items():
        for f, pos in enumerate(positions):
            tracks["players"][f][track_id] = {"bbox": [0, 0, 1, 1], "team": team,
                                              "position_m": list(pos),
                                              "global_id": track_id}
    return tracks


def test_possession_team_disputed_and_free_keeps_last():
    n = 40
    ball = [(10, 10)] * 10 + [(20, 10)] * 10 + [(30, 10)] * 10 + [(40, 10)] * 10
    players = {
        1: (0, [(10.5, 10)] * n),              # equipa 0 junto à bola no início
        2: (1, [(40.5, 10)] * n),              # equipa 1 junto à bola no fim
        3: (0, [(20.4, 10)] * n),              # disputa em x=20
        4: (1, [(19.6, 10)] * n),
    }
    tracks = tracks_with(ball, players)
    game = Game.from_tracks(tracks, 25)
    pa = PossessionAssigner(min_frames=3)
    possession = pa.assign(game, tracks["ball"])
    assert possession[5] == 0
    assert possession[15] == DISPUTED
    assert possession[25] == DISPUTED   # bola livre em x=30: mantém o anterior
    assert possession[35] == 1
    assert game.possession == possession
    assert game.ball_holder[5] == 1 and game.ball_holder[15] is None
    assert [p.has_ball for p in game.positions_at(5) if p.global_id == 1] == [True]
    stats = pa.statistics()
    assert abs(sum(stats.values()) - 100) < 1e-6


def test_possession_ignores_short_flicker():
    n = 30
    ball = [(10, 10)] * n
    team0 = [(10.5, 10)] * n
    team1 = [(30, 10)] * 14 + [(10.2, 10)] * 2 + [(30, 10)] * 14  # 2 frames perto
    tracks = tracks_with(ball, {1: (0, team0), 2: (1, team1)})
    possession = PossessionAssigner(min_frames=5).assign(
        Game.from_tracks(tracks, 25), tracks["ball"])
    assert set(possession) == {0}


def test_no_ball_means_no_possession():
    tracks = tracks_with([None] * 5, {1: (0, [(10, 10)] * 5)})
    assert PossessionAssigner().assign(Game.from_tracks(tracks, 25), tracks["ball"]) == [None] * 5


def test_long_gap_interpolation_does_not_change_possession():
    n = 30
    ball = [(10, 10)] * n
    tracks = tracks_with(ball, {1: (0, [(10.5, 10)] * 10 + [(40, 10)] * 20),
                                2: (1, [(40, 10)] * 10 + [(10.5, 10)] * 20)})
    for f in range(10, n):  # lacuna longa, interpolada, junto ao jogador 2
        tracks["ball"][f][1].update(interpolated=True, gap_frames=20)
    possession = PossessionAssigner(min_frames=3, max_trusted_gap=8).assign(
        Game.from_tracks(tracks, 25), tracks["ball"])
    assert set(possession) == {0}
