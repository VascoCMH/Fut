import cv2

def read_video(video_path):
    vid=cv2.VideoCapture(video_path)
    fps=vid.get(cv2.CAP_PROP_FPS)
    frames=[]
    while True:
        ret, frame=vid.read()
        if not ret:
            break
        frames.append(frame)
    return fps, frames


def save_video(frames, output_path, fps):
    fourcc=cv2.VideoWriter_fourcc(*'mp4v')
    out=cv2.VideoWriter(output_path, fourcc, fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(frame)
    out.release()
    return output_path

import cv2
import numpy as np


def camera_moved(frame1, frame2, threshold=0.35, width=640):

    # Se os frames forem inválidos, assumir movimento.
    if frame1 is None or frame2 is None:
        return True

    if frame1.shape[:2] != frame2.shape[:2]:
        return True

    # Reduzir resolução e converter para grayscale.
    def prepare(frame):
        h, w = frame.shape[:2]
        scale = min(1.0, width / w)

        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA
            )

        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        return frame

    prev = prepare(frame1)
    curr = prepare(frame2)

    # Frames iguais -> nenhuma alteração observável.
    if np.array_equal(prev, curr):
        return False

    # 1. Encontrar pontos visuais no frame anterior.
    points = cv2.goodFeaturesToTrack(
        prev,
        maxCorners=250,
        qualityLevel=0.01,
        minDistance=8
    )

    if points is None or len(points) < 20:
        return True

    # 2. Acompanhar os pontos no frame seguinte.
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        prev, curr, points, None,
        winSize=(17, 17),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS |
            cv2.TERM_CRITERIA_COUNT,
            20, 0.03
        )
    )

    if tracked is None or status is None:
        return True

    valid = status.ravel().astype(bool)

    src = points.reshape(-1, 2)[valid]
    dst = tracked.reshape(-1, 2)[valid]

    if len(src) < 20:
        return True

    # 3. Estimar movimento global com RANSAC.
    M, mask = cv2.estimateAffinePartial2D(
        src, dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=1.5,
        maxIters=300,
        confidence=0.97,
        refineIters=3
    )

    if M is None or mask is None:
        return True

    if not np.isfinite(M).all():
        return True

    inliers = mask.ravel().astype(bool)

    # 4. Rejeitar estimativas pouco fiáveis.
    if inliers.sum() < 15 or inliers.mean() < 0.5:
        return True

    h, w = prev.shape

    # Os pontos fiáveis devem estar distribuídos
    # por pelo menos 3 regiões da imagem.
    cells = np.column_stack((
        np.minimum((src[inliers, 0] * 3 / w).astype(int), 2),
        np.minimum((src[inliers, 1] * 3 / h).astype(int), 2)
    ))

    if len(np.unique(cells, axis=0)) < 3:
        return True

    # 5. Medir deslocação global em 9 locais da imagem.
    test_points = np.array([
        [0, 0],       [w / 2, 0],       [w - 1, 0],
        [0, h / 2],   [w / 2, h / 2],   [w - 1, h / 2],
        [0, h - 1],   [w / 2, h - 1],   [w - 1, h - 1]
    ], dtype=np.float64)

    transformed = test_points @ M[:, :2].T + M[:, 2]

    displacement = np.linalg.norm(
        transformed - test_points,
        axis=1
    )

    # Movimento relevante ou ausência de movimento.
    return bool(displacement.max() >= threshold)