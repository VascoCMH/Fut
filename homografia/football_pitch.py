
import numpy as np

class FootballPitch:

    def __init__(self):
        # Dimensões originais Roboflow em centímetros
        self.length = 12000
        self.width = 7000

        # IDs originais dos keypoints
        self.keypoints = {
            # Linha de fundo esquerda
            1:  (0, 0),
            2:  (0, 1450),
            3:  (0, 2584),
            4:  (0, 4416),
            5:  (0, 5550),
            6:  (0, 7000),

            # Pequena área esquerda
            7:  (550, 2584),
            8:  (550, 4416),

            # Marca de penálti esquerda
            9:  (1100, 3500),

            # Grande área esquerda
            10: (2015, 1450),
            11: (2015, 2584),
            12: (2015, 4416),
            13: (2015, 5550),

            # Meio-campo
            14: (6000, 0),
            15: (6000, 2585),
            16: (6000, 4415),
            17: (6000, 7000),

            # Grande área direita
            18: (9985, 1450),
            19: (9985, 2584),
            20: (9985, 4416),
            21: (9985, 5550),

            # Marca de penálti direita
            22: (10900, 3500),

            # Pequena área direita
            23: (11450, 2584),
            24: (11450, 4416),

            # Linha de fundo direita
            25: (12000, 0),
            26: (12000, 1450),
            27: (12000, 2584),
            28: (12000, 4416),
            29: (12000, 5550),
            30: (12000, 7000),

            # Círculo central
            31: (5085, 3500),
            32: (6915, 3500)
        }

    def get_keypoints(self):
        # Devolver coordenadas em metros
        return {
            key: (x / 100, y / 100)
            for key, (x, y) in self.keypoints.items()
        }

    def get_vertices(self):
        # Array na ordem esperada pela Roboflow
        return np.array(
            list(self.keypoints.values()),
            dtype=np.float32
        )
