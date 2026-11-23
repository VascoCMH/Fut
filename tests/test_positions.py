"""Testes sintéticos da estabilização da homografia e da suavização."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from positions import HomographyStabilizer, PositionSmoother, hampel_outliers  # noqa: E402

FPS = 25
PX_PER_M = 10.0
# Homografia "verdadeira": píxeis -> metros (escala simples, sem perspetiva).
TRUE_MATRIX = np.array([[1 / PX_PER_M, 0, 0], [0, 1 / PX_PER_M, 0], [0, 0, 1.0]])


def to_px(pos_m):
    return np.asarray(pos_m) * PX_PER_M


def bbox_at(pos_m, height=40):
    x, y = to_px(pos_m)
    return [x - 8, y - height, x + 8, y]


def make_tracks(n_frames, people, matrices=None):
    """people: {track_id: (global_id, team, função frame -> posição em m ou None)}"""
    tracks = {"players": [{} for _ in range(n_frames)],
              "referees": [{} for _ in range(n_frames)],
              "ball": [{} for _ in range(n_frames)]}
    for track_id, (gid, team, path) in people.items():
        for f in range(n_frames):
            pos = path(f)
            if pos is None:
                continue
            bbox = bbox_at(pos)
            m = TRUE_MATRIX if matrices is None else matrices[f]
            x1, _, x2, y2 = bbox
            p = m @ np.array([(x1 + x2) / 2, y2, 1.0])
            tracks["players"][f][track_id] = {
                "bbox": bbox, "position_m": (p[:2] / p[2]).tolist(),
                "global_id": gid, "team": team, "role": "player",
                "team_color": (255, 255, 255)}
    return tracks


def linear(start, velocity, frames=None):
    def path(f):
        if frames is not None and f not in frames:
            return None
        return np.asarray(start, float) + np.asarray(velocity, float) * f / FPS
    return path


# ---------------------------------------------------------- estabilização
def test_stabilizer_removes_common_jump():
    n = 100
    matrices = [TRUE_MATRIX.copy() for _ in range(n)]
    shift = np.array([[1, 0, 3.0], [0, 1, -2.0], [0, 0, 1]])  # erro de 3 m e 2 m
    matrices[50] = shift @ TRUE_MATRIX
    people = {t: (t, 0, linear((10 + 7 * t, 20 + 3 * t), (2, 1))) for t in range(1, 9)}
    tracks = make_tracks(n, people, matrices)
    fixed = HomographyStabilizer(windows=(12, 6)).stabilize(tracks, matrices)
    feet = np.array([(bbox_at(people[1][2](50))[0] + 8), bbox_at(people[1][2](50))[3], 1.0])
    p = fixed[50] @ feet
    assert np.allclose(p[:2] / p[2], people[1][2](50), atol=0.1)
    # Frames corretos ficam praticamente iguais.
    assert np.allclose(fixed[20], TRUE_MATRIX, atol=1e-3)


# ------------------------------------------------------------- suavização
def test_hampel_flags_isolated_spike():
    frames = np.arange(30)
    points = np.column_stack([frames * 0.2, np.zeros(30)])
    points[15] += [5, 5]
    bad = hampel_outliers(frames, points)
    assert bad[15] and bad.sum() == 1


def test_constant_speed_is_recovered():
    tracks = make_tracks(100, {1: (1, 0, linear((10, 20), (3, 4)))})  # 5 m/s
    PositionSmoother(fps=FPS).smooth(tracks)
    speeds = [tracks["players"][f][1]["speed_ms"] for f in range(100)]
    assert np.allclose(speeds, 5.0, atol=0.05)
    reliable = [tracks["players"][f][1]["speed_reliable"] for f in range(100)]
    assert not reliable[0] and not reliable[99] and all(reliable[12:88])


def test_spike_is_removed_from_smoothed_path():
    path = linear((10, 20), (2, 0))
    tracks = make_tracks(60, {1: (1, 0, path)})
    tracks["players"][30][1]["position_m"] = [40.0, 45.0]  # salto de ~30 m
    PositionSmoother(fps=FPS).smooth(tracks)
    person = tracks["players"][30][1]
    assert person["position_quality"] == "outlier"
    assert np.allclose(person["position_smooth"], path(30), atol=0.05)
    assert max(tracks["players"][f][1]["speed_ms"] for f in range(60)) < 2.1


def test_gap_between_linked_tracks_is_filled_in_straight_line():
    # O mesmo jogador (global_id 7) em duas tracks do ByteTrack, com 30 frames
    # sem ser visto entre elas.
    tracks = make_tracks(130, {
        1: (7, 0, linear((10, 20), (5, 0), frames=range(0, 50))),
        2: (7, 0, linear((10, 20), (5, 0), frames=range(80, 130))),
    })
    PositionSmoother(fps=FPS).smooth(tracks)
    for f in range(50, 80):
        entry = tracks["estimated"][f][7]
        assert entry["estimated"] and entry["gap_frames"] == 30
        assert np.allclose(entry["position_smooth"], (10 + 5 * f / FPS, 20), atol=0.05)
    assert 7 not in tracks["estimated"][20] and 7 not in tracks["estimated"][100]


def test_max_gap_limits_estimation():
    tracks = make_tracks(200, {
        1: (7, 0, linear((10, 20), (1, 0), frames=range(0, 50))),
        2: (7, 0, linear((10, 20), (1, 0), frames=range(150, 200))),
    })
    PositionSmoother(fps=FPS, max_gap_s=2.0).smooth(tracks)  # ausência de 4 s
    assert all(7 not in tracks["estimated"][f] for f in range(50, 150))


def test_boxes_cut_by_the_border_are_not_trusted():
    path = linear((10, 20), (2, 0))
    tracks = make_tracks(60, {1: (1, 0, path)})
    # Frame 30: caixa cortada na borda esquerda e posição errada.
    tracks["players"][30][1]["bbox"][0] = 0
    tracks["players"][30][1]["position_m"] = [13.0, 23.0]
    PositionSmoother(fps=FPS, frame_size=(1920, 1080)).smooth(tracks)
    person = tracks["players"][30][1]
    assert person["position_quality"] == "border"
    assert np.allclose(person["position_smooth"], path(30), atol=0.05)


def test_track_with_only_border_boxes_still_anchors_a_gap():
    # Guarda-redes visto só colado à borda (frames 0-14) e de novo 4 s depois.
    tracks = make_tracks(130, {
        1: (20, 0, linear((10, 20), (0, 0), frames=range(0, 15))),
        2: (20, 0, linear((10, 20), (0, 0), frames=range(115, 130))),
    })
    for f in range(0, 15):
        tracks["players"][f][1]["bbox"][0] = 0
    PositionSmoother(fps=FPS, frame_size=(1920, 1080)).smooth(tracks)
    assert all(20 in tracks["estimated"][f] for f in range(15, 115))
