"""Testes sintéticos do ReID (não precisam de vídeo nem de modelos).

Correr com:  python -m pytest tests/
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reid import ReIdentifier  # noqa: E402

FPS = 25


def empty_tracks(n_frames):
    return {"players": [{} for _ in range(n_frames)],
            "referees": [{} for _ in range(n_frames)],
            "ball": [{} for _ in range(n_frames)]}


def add(tracks, track_id, frames, start_m, velocity_ms, team=0, role="player",
        with_position=True, category="players"):
    """Pessoa em movimento retilíneo; bbox em píxeis (~20 px por metro)."""
    for f in frames:
        t = f / FPS
        x, y = np.asarray(start_m) + np.asarray(velocity_ms) * t
        px, py = 100 + 20 * x, 100 + 20 * y
        tracks[category][f][track_id] = {
            "bbox": [px - 10, py - 36, px + 10, py],
            "position_m": [float(x), float(y)] if with_position else None,
            "team": team, "role": role,
        }


def test_rejoins_track_split_by_occlusion():
    tracks = empty_tracks(200)
    add(tracks, 1, range(0, 80), (10, 20), (3, 0))
    add(tracks, 2, range(120, 200), (10, 20), (3, 0))  # mesmo jogador, 1,6 s depois
    reid = ReIdentifier(fps=FPS)
    reid.assign(tracks)
    assert reid.links == {1: 2}
    assert tracks["players"][0][1]["global_id"] == tracks["players"][150][2]["global_id"]


def test_does_not_swap_players_crossing():
    # Dois colegas de equipa cruzam-se e desaparecem juntos; ao reaparecer,
    # cada um continua na sua direção.
    tracks = empty_tracks(200)
    add(tracks, 1, range(0, 90), (20, 30), (4, 0))     # vai para a direita
    add(tracks, 2, range(0, 90), (44, 30.5), (-4, 0))  # vai para a esquerda
    add(tracks, 3, range(110, 200), (20, 30), (4, 0))   # continuação do 1
    add(tracks, 4, range(110, 200), (44, 30.5), (-4, 0))  # continuação do 2
    reid = ReIdentifier(fps=FPS).fit(tracks)
    assert reid.links.get(1) != 4 and reid.links.get(2) != 3
    assert reid.links == {1: 3, 2: 4}


def test_ambiguous_case_is_left_unlinked():
    # Dois colegas parados lado a lado desaparecem e reaparecem: não há forma
    # de saber quem é quem só pela posição, por isso não se liga.
    tracks = empty_tracks(200)
    add(tracks, 1, range(0, 80), (30, 30), (0, 0))
    add(tracks, 2, range(0, 80), (30, 31), (0, 0))
    add(tracks, 3, range(130, 200), (30, 30.5), (0, 0))
    add(tracks, 4, range(130, 200), (30, 30.5), (0, 0))
    reid = ReIdentifier(fps=FPS).fit(tracks)
    assert reid.links == {}
    assert reid.rejected >= 1


def test_never_links_different_teams_or_roles():
    tracks = empty_tracks(200)
    add(tracks, 1, range(0, 80), (10, 20), (3, 0), team=0)
    add(tracks, 2, range(100, 200), (10, 20), (3, 0), team=1)
    add(tracks, 3, range(0, 80), (5, 35), (0, 0), team=0, role="goalkeeper")
    add(tracks, 4, range(100, 200), (5, 35), (0, 0), team=0)
    reid = ReIdentifier(fps=FPS).fit(tracks)
    assert reid.links == {}


def test_never_links_overlapping_tracks():
    tracks = empty_tracks(200)
    add(tracks, 1, range(0, 100), (10, 20), (3, 0))
    add(tracks, 2, range(90, 200), (10, 20), (3, 0))
    assert ReIdentifier(fps=FPS).fit(tracks).links == {}


def test_rejects_impossible_distance():
    tracks = empty_tracks(200)
    add(tracks, 1, range(0, 80), (10, 20), (0, 0))
    add(tracks, 2, range(105, 200), (60, 20), (0, 0))  # 50 m em 1 s
    assert ReIdentifier(fps=FPS).fit(tracks).links == {}


def test_pixel_fallback_only_for_short_gaps():
    tracks = empty_tracks(400)
    add(tracks, 1, range(0, 80), (10, 20), (2, 0), with_position=False)
    add(tracks, 2, range(95, 200), (10, 20), (2, 0), with_position=False)   # 0,6 s
    add(tracks, 3, range(260, 400), (10, 20), (2, 0), with_position=False)  # 2,4 s
    reid = ReIdentifier(fps=FPS).fit(tracks)
    assert reid.links == {1: 2}


def test_referees_link_among_themselves():
    tracks = empty_tracks(200)
    add(tracks, 7, range(0, 80), (50, 30), (1, 0), team=None, role="referee",
        category="referees")
    add(tracks, 8, range(110, 200), (50, 30), (1, 0), team=None, role="referee",
        category="referees")
    reid = ReIdentifier(fps=FPS)
    reid.assign(tracks)
    assert reid.links == {7: 8}
    assert tracks["referees"][0][7]["global_id"] == tracks["referees"][150][8]["global_id"]


def test_goalkeeper_links_after_long_absence():
    # A câmara está do outro lado do campo durante 19 s.
    tracks = empty_tracks(750)
    add(tracks, 20, range(0, 15), (104, 40), (0, 0), role="goalkeeper")
    add(tracks, 99, range(494, 750), (112, 43), (0, 0), role="goalkeeper")
    add(tracks, 1, range(0, 15), (60, 30), (0, 0))
    add(tracks, 2, range(494, 750), (60, 30), (0, 0))  # jogador: 19 s é demasiado
    reid = ReIdentifier(fps=FPS).fit(tracks)
    assert reid.links == {20: 99}
