"""Transformação entre píxeis da câmara e coordenadas do campo em metros."""

import cv2
import numpy as np

from .football_pitch import FootballPitch


class Homography:
    """Calibra uma imagem com os keypoints do campo (IDs de 1 a 32)."""

    def __init__(self, pitch=None, model_path="modelo/pitch.pt"):
        self.pitch = pitch if pitch is not None else FootballPitch()
        self.model_path = model_path
        self.model = None
        self.matrix = None
        self.inlier_ids = []

    def get_matrix(self, frame):
        """Deteta os keypoints do frame e devolve a matriz píxeis -> metros.

        O modelo é carregado na primeira chamada e reutilizado nas seguintes.
        Lança ValueError se não for possível calibrar este frame.
        """
        self.matrix = None
        self.inlier_ids = []
        if self.model is None:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)
        result = self.model.predict(frame, verbose=False)[0]
        self.fit_yolo(result)
        return self.matrix.copy()

    @staticmethod
    def update_tracks(matrix, tracks):
        """Atualiza e devolve as tracks de UM frame, preservando IDs e dados.

        Formato: {categoria: {track_id: {"bbox": [x1, y1, x2, y2], ...}}}.
        Acrescenta feet_px e position_m a cada objeto, no próprio dicionário.
        matrix=None limpa position_m, evitando reutilizar posições anteriores.
        """
        if matrix is not None:
            matrix = np.asarray(matrix, dtype=np.float64)
            if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
                    or np.linalg.matrix_rank(matrix) < 3):
                raise ValueError("A matriz deve ser 3 × 3, finita e invertível.")
        for objects in tracks.values():
            for track in objects.values():
                bbox = np.asarray(track["bbox"], dtype=np.float64)
                if bbox.shape != (4,) or not np.isfinite(bbox).all():
                    raise ValueError("Cada bbox deve conter quatro valores finitos.")
                x1, _, x2, y2 = bbox
                feet = [float((x1 + x2) / 2), float(y2)]
                track["feet_px"] = feet
                track["position_m"] = None
                if matrix is None:
                    continue
                projected = matrix @ np.array([*feet, 1.0])
                if not np.isfinite(projected).all() or np.isclose(projected[2], 0):
                    continue
                position = projected[:2] / projected[2]
                if np.isfinite(position).all():
                    track["position_m"] = position.tolist()
        return tracks

    def fit(self, image_keypoints, ransac_threshold=5.0):
        """Recebe {ID: (x_px, y_px)}; o limiar RANSAC é em píxeis.

        São necessários pelo menos quatro pares numa configuração não
        degenerada. Recalibrar quando a câmara mudar de posição ou zoom.
        """
        self.matrix = None
        self.inlier_ids = []
        if not np.isfinite(ransac_threshold) or ransac_threshold <= 0:
            raise ValueError("O limiar RANSAC deve ser positivo e finito.")
        field_keypoints = self.pitch.get_keypoints()
        ids = list(image_keypoints)
        if any(key not in field_keypoints for key in ids):
            raise ValueError("Os IDs dos keypoints devem estar entre 1 e 32.")
        if len(ids) < 4:
            raise ValueError("São necessários pelo menos quatro keypoints válidos.")
        source = self._points([image_keypoints[key] for key in ids])
        target = self._points([field_keypoints[key] for key in ids])
        for points in (source, target):
            if np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
                raise ValueError("Os keypoints não podem estar todos na mesma linha.")

        # Estimar campo -> imagem para medir o erro RANSAC em píxeis.
        inverse, mask = cv2.findHomography(
            target, source, cv2.RANSAC, ransac_threshold
        )
        if inverse is None or mask is None or int(mask.sum()) < 4:
            raise ValueError("Não foi possível estimar uma homografia válida.")
        if not np.isfinite(inverse).all() or np.linalg.matrix_rank(inverse) < 3:
            raise ValueError("A homografia estimada é degenerada.")
        self.matrix = np.linalg.inv(inverse)
        self.inlier_ids = [key for key, valid in zip(ids, mask.ravel()) if valid]
        return self

    def fit_yolo(self, result, confidence_threshold=0.5, ransac_threshold=5.0):
        """Calibra com um Results do Ultralytics usando coordenadas em píxeis.

        O índice 0 do modelo corresponde ao ID 1 do FootballPitch, etc.
        O modelo tem de usar exatamente esta ordem dos 32 keypoints.
        """
        self.matrix = None
        self.inlier_ids = []
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("O limiar de confiança deve estar entre 0 e 1.")
        keypoints = result.keypoints
        if keypoints is None or len(keypoints.xy) == 0:
            raise ValueError("O modelo não detetou keypoints do campo.")
        xy = keypoints.xy[0].detach().cpu().numpy()
        if len(xy) != len(self.pitch.keypoints):
            raise ValueError("O modelo deve devolver os 32 keypoints do campo.")
        confidence = keypoints.conf
        scores = (
            confidence[0].detach().cpu().numpy()
            if confidence is not None else np.ones(len(xy))
        )
        visible = {
            index + 1: point
            for index, (point, score) in enumerate(zip(xy, scores))
            if np.isfinite(point).all() and np.isfinite(score)
            and score >= confidence_threshold and not np.all(point == 0)
        }
        return self.fit(visible, ransac_threshold)

    @staticmethod
    def _points(points):
        array = np.asarray(points, dtype=np.float64)
        if array.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2 or not np.isfinite(array).all():
            raise ValueError("Os pontos devem ter a forma (N, 2) e valores finitos.")
        return array

    def transform_points(self, points, inverse=False):
        """Converte (N, 2) píxeis -> metros; inverse=True faz metros -> píxeis."""
        if self.matrix is None:
            raise ValueError("É necessário calibrar a homografia primeiro.")
        points = self._points(points)
        if len(points) == 0:
            return points
        matrix = np.linalg.inv(self.matrix) if inverse else self.matrix
        homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
        if np.any(np.isclose(homogeneous[:, 2], 0)):
            raise ValueError("Há pontos cuja projeção está no infinito.")
        return homogeneous[:, :2] / homogeneous[:, 2:3]
