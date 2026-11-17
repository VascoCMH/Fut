"""Painel de posse de bola (percentagens acumuladas até ao frame atual)."""

import cv2

from .possession import DISPUTED

DISPUTED_COLOR = (150, 150, 150)
TEXT = (255, 255, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_possession(frame, counts, current, team_colors, origin=(20, 20)):
    """counts: Counter da posse desde o início até este frame (0, 1,
    "disputed"); current: posse neste frame. Recebe contagens acumuladas em
    vez da lista completa para não ser O(n²) em vídeos longos."""
    total = max(sum(counts.get(s, 0) for s in (0, 1, DISPUTED)), 1)
    states = [0, 1, DISPUTED]
    shares = {s: counts.get(s, 0) / total for s in states}
    colors = {0: team_colors.get(0, DISPUTED_COLOR), 1: team_colors.get(1, DISPUTED_COLOR),
              DISPUTED: DISPUTED_COLOR}
    names = {0: "Equipa 0", 1: "Equipa 1", DISPUTED: "Disputada"}

    x0, y0 = origin
    width, height = 420, 112
    # Fundo escuro semitransparente (só na zona do painel).
    roi = frame[y0:y0 + height, x0:x0 + width]
    roi[:] = (0.25 * roi + 0.75 * 30).astype(roi.dtype)

    now = "sem bola" if current is None else names[current]
    cv2.putText(frame, f"Posse de bola  |  agora: {now}", (x0 + 12, y0 + 26),
                FONT, 0.6, TEXT, 2, cv2.LINE_AA)

    # Barra dividida pelas três partes.
    bar_x, bar_y, bar_w, bar_h = x0 + 12, y0 + 40, width - 24, 20
    x = bar_x
    for state in states:
        w = int(round(shares[state] * bar_w))
        cv2.rectangle(frame, (x, bar_y), (x + w, bar_y + bar_h), colors[state], -1)
        x += w
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), TEXT, 1)

    # Legenda com quadrado de cor e percentagem.
    x = x0 + 12
    for state in states:
        cv2.rectangle(frame, (x, y0 + 80), (x + 14, y0 + 94), colors[state], -1)
        cv2.rectangle(frame, (x, y0 + 80), (x + 14, y0 + 94), TEXT, 1)
        label = f"{names[state]} {100 * shares[state]:.0f}%"
        cv2.putText(frame, label, (x + 20, y0 + 93), FONT, 0.5, TEXT, 1, cv2.LINE_AA)
        x += 135
    return frame
