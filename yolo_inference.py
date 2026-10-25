import argparse
import math
from pathlib import Path

import cv2
from ultralytics import YOLO
from homografia import Homography
from homografia.minimap import PitchMinimap


def main():
    parser = argparse.ArgumentParser(description="Testar homografia num vídeo.")
    parser.add_argument("--source", default="input_videos/inputvideo.mp4")
    parser.add_argument("--output", default="output_videos/pitch_homography.mp4")
    parser.add_argument("--model", default="modelo/pitch.pt")
    parser.add_argument("--player-model", default="modelo/player.pt")
    args = parser.parse_args()
    if Path(args.source).resolve() == Path(args.output).resolve():
        parser.error("O vídeo de saída deve ser diferente do vídeo de entrada.")

    model = YOLO(args.model)
    player_model = YOLO(args.player_model)
    player_classes = [
        class_id for class_id, name in player_model.names.items()
        if name.lower() in {"player", "goalkeeper"}
    ]
    if not player_classes:
        parser.error("O modelo de jogadores deve ter classes player/goalkeeper.")
    capture = cv2.VideoCapture(args.source)
    writer = None
    homography = Homography()
    minimap = PitchMinimap(homography.pitch)
    frame_count = 0
    calibrated_count = 0
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo: {args.source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0
            print("FPS indisponível; a usar 25 FPS no vídeo de saída.")
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_count += 1
            result = model.predict(frame, verbose=False)[0]
            player_result = player_model.predict(
                frame, classes=player_classes, conf=0.3, verbose=False
            )[0]
            annotated_frame = result.plot()
            players = []
            try:
                # Calibrar novamente em cada frame, acompanhando movimento/zoom.
                homography.fit_yolo(result)
            except ValueError as error:
                status = "Homografia indisponivel"
                color = (0, 0, 255)
                if frame_count == 1 or frame_count % 100 == 0:
                    print(f"Frame {frame_count}: {error}")
            else:
                calibrated_count += 1
                status = f"Homografia OK: {len(homography.inlier_ids)} pontos"
                color = (0, 255, 0)

            # O mesmo número identifica o jogador no vídeo e no minimapa.
            # Estes números são locais ao frame, não são IDs de tracking.
            for number, box in enumerate(player_result.boxes.xyxy.cpu().numpy(), 1):
                x1, y1, x2, y2 = box
                feet = [(float(x1) + float(x2)) / 2, float(y2)]
                foot_pixel = (round(feet[0]), round(feet[1]))
                cv2.rectangle(annotated_frame, (round(x1), round(y1)),
                              (round(x2), round(y2)), (0, 220, 255), 1)
                cv2.circle(annotated_frame, foot_pixel, 4, (0, 220, 255), -1)
                cv2.putText(annotated_frame, f"P{number}",
                            (round(x1), max(15, round(y1) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2)
                if homography.matrix is not None:
                    try:
                        position = homography.transform_points([feet])[0]
                    except ValueError:
                        continue
                    players.append((number, position))

            minimap.overlay(annotated_frame, players, homography.matrix is not None)

            cv2.putText(
                annotated_frame, f"Frame {frame_count} | {status}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
            )
            if writer is None:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                height, width = annotated_frame.shape[:2]
                writer = cv2.VideoWriter(
                    args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Não foi possível criar o vídeo: {args.output}")
            writer.write(annotated_frame)
            if frame_count % 100 == 0:
                print(f"{frame_count} frames processados; {calibrated_count} calibrados.")
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if frame_count == 0:
        raise RuntimeError("O vídeo não contém frames legíveis.")
    print(f"Vídeo guardado em: {args.output}")
    print(f"Homografia calculada em {calibrated_count}/{frame_count} frames.")


if __name__ == "__main__":
    main()
