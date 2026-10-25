"""Minimapa para comparar posições projetadas com o vídeo original."""

import cv2
import numpy as np


class PitchMinimap:
    def __init__(self, pitch):
        self.points = pitch.get_keypoints()
        self.length = pitch.length / 100
        self.width = pitch.width / 100

    def overlay(self, frame, players, calibrated):
        """players contém pares (número no frame, posição em metros).

        Pontos fora do campo são contados, sem os prender às linhas do campo.
        """
        height, width = frame.shape[:2]
        panel_width = min(440, width // 2)
        margin = 18
        scale = (panel_width - 2 * margin) / self.length
        field_height = round(self.width * scale)
        panel_height = field_height + 85
        # Reduzir o painel para também caber em vídeos de pouca altura.
        if panel_height > height - 20:
            scale = max(0.01, (height - 105) / self.width)
            panel_width = round(self.length * scale) + 2 * margin
            field_height = round(self.width * scale)
            panel_height = field_height + 85
        if panel_width < 100 or panel_height > height or panel_width > width:
            return frame
        panel = np.full((panel_height, panel_width, 3), (35, 85, 35), np.uint8)
        top = 35

        def pixel(point):
            x, y = point
            return round(margin + x * scale), round(top + y * scale)

        edges = [
            (1, 25), (25, 30), (30, 6), (6, 1), (14, 17),
            (2, 10), (10, 13), (13, 5),
            (3, 7), (7, 8), (8, 4),
            (26, 18), (18, 21), (21, 29),
            (27, 23), (23, 24), (24, 28),
        ]
        for start, end in edges:
            cv2.line(panel, pixel(self.points[start]), pixel(self.points[end]),
                     (220, 220, 220), 1, cv2.LINE_AA)
        center = (self.length / 2, self.width / 2)
        radius = abs(self.points[32][0] - self.points[31][0]) / 2
        cv2.circle(panel, pixel(center), round(radius * scale), (220, 220, 220), 1)
        for point in (center, self.points[9], self.points[22]):
            cv2.circle(panel, pixel(point), 2, (220, 220, 220), -1)

        outside = 0
        for number, position in players:
            x, y = position
            if not np.isfinite(position).all() or not (
                0 <= x <= self.length and 0 <= y <= self.width
            ):
                outside += 1
                continue
            location = pixel(position)
            cv2.circle(panel, location, 5, (0, 0, 0), -1)
            cv2.circle(panel, location, 4, (0, 220, 255), -1)
            cv2.putText(panel, str(number), (location[0] + 5, location[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        title = "Minimapa | metros" if calibrated else "Minimapa | sem calibracao"
        cv2.putText(panel, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1)
        footer = f"Projetados: {len(players)} | Fora: {outside}"
        cv2.putText(panel, footer, (8, panel_height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        x0 = max(0, width - panel_width - 10)
        y0 = max(0, height - panel_height - 10)
        frame[y0:y0 + panel_height, x0:x0 + panel_width] = panel
        return frame
