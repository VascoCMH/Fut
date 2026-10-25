import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from homografia import Homography
from homografia.minimap import PitchMinimap
from trackers import Tracker


def finish_phase(phase, started):
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
    elapsed = perf_counter() - started
    finished_at = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    print(f"{phase} | fim {finished_at} UTC | duracao {elapsed:.3f} s", flush=True)


def read_frames(source):
    """Reabrir o vídeo por passagem, sem guardar imagens de todo o vídeo."""
    capture = cv2.VideoCapture(source)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo: {source}")
        while True:
            success, frame = capture.read()
            if not success:
                break
            yield frame
    finally:
        capture.release()


def track_video(source, model_path):
    """Passagem 1: usar o ByteTrack existente e guardar IDs/caixas por frame."""
    tracker = Tracker(model_path)
    records = []
    batch = []

    def process_batch():
        # A mesma instância do tracker mantém o estado entre lotes.
        tracks = tracker.get_tracks(batch)
        for players in tracks["players"]:
            records.append({
                "frame_index": len(records),
                "players": [
                    {"track_id": int(track_id), "bbox": list(map(float, player["bbox"]))}
                    for track_id, player in players.items()
                ],
            })
        print(f"Tracking: {len(records)} frames analisados.", flush=True)

    for frame in read_frames(source):
        batch.append(frame)
        if len(batch) == 32:
            process_batch()
            batch.clear()
    if batch:
        process_batch()
    if not records:
        raise RuntimeError("O vídeo não contém frames legíveis.")
    return records


def project_video(source, model_path, records):
    """Passagem 2: detetar o campo e acrescentar posições em metros à lista."""
    model = YOLO(model_path)
    homography = Homography()
    count = 0
    for index, frame in enumerate(read_frames(source)):
        if index >= len(records):
            raise RuntimeError("O número de frames mudou entre passagens.")
        record = records[index]
        result = model.predict(frame, verbose=False)[0]
        try:
            homography.fit_yolo(result)
        except ValueError as error:
            record["calibration_error"] = str(error)
        else:
            record["calibration_error"] = None
        record["homography"] = (
            homography.matrix.tolist() if homography.matrix is not None else None
        )
        record["inlier_ids"] = list(map(int, homography.inlier_ids))
        for player in record["players"]:
            x1, _, x2, y2 = player["bbox"]
            feet = [(x1 + x2) / 2, y2]
            player["feet_px"] = feet
            player["position_m"] = None
            if homography.matrix is not None:
                try:
                    position = homography.transform_points([feet])[0]
                except ValueError:
                    continue
                if np.isfinite(position).all():
                    player["position_m"] = position.tolist()
        count += 1
        if count % 100 == 0:
            print(f"Homografia e posições: {count}/{len(records)} frames.", flush=True)
    if count != len(records):
        raise RuntimeError("O número de frames mudou entre passagens.")
    return homography.pitch


def render_video(source, output, records, pitch, fps):
    """Passagem 3: desenhar exclusivamente a partir dos dados calculados."""
    minimap = PitchMinimap(pitch)
    writer = None
    count = 0
    try:
        for index, frame in enumerate(read_frames(source)):
            if index >= len(records):
                raise RuntimeError("O número de frames mudou entre passagens.")
            record = records[index]
            calibrated = record["homography"] is not None
            projected = []
            for player in record["players"]:
                track_id = player["track_id"]
                x1, y1, x2, y2 = map(round, player["bbox"])
                feet = tuple(map(round, player["feet_px"]))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 1)
                cv2.circle(frame, feet, 4, (0, 220, 255), -1)
                cv2.putText(frame, f"ID {track_id}", (x1, max(15, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2)
                if player["position_m"] is not None:
                    projected.append((track_id, player["position_m"]))
            minimap.overlay(frame, projected, calibrated)
            status = (
                f"Homografia calculada: {len(record['inlier_ids'])} pontos"
                if calibrated else "Homografia indisponivel"
            )
            cv2.putText(frame, f"Frame {index + 1} | {status}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if calibrated else (0, 0, 255), 2)
            if writer is None:
                Path(output).parent.mkdir(parents=True, exist_ok=True)
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Não foi possível criar o vídeo: {output}")
            writer.write(frame)
            count += 1
            if count % 100 == 0:
                print(f"Desenho: {count}/{len(records)} frames.", flush=True)
    finally:
        if writer is not None:
            writer.release()
    if count != len(records):
        raise RuntimeError("O número de frames mudou entre passagens.")


def main():
    parser = argparse.ArgumentParser(description="Tracking, homografia e desenho por fases.")
    parser.add_argument("--source", default="input_videos/inputvideo.mp4")
    parser.add_argument("--output", default="output_videos/pitch_homography.mp4")
    parser.add_argument("--model", default="modelo/pitch.pt")
    parser.add_argument("--player-model", default="modelo/player.pt")
    parser.add_argument("--positions", help="Ficheiro JSON; por defeito, junto do vídeo de saída.")
    args = parser.parse_args()
    positions_path = Path(args.positions) if args.positions else Path(args.output).with_suffix(".json")
    paths = [Path(args.source), Path(args.output), positions_path,
             Path(args.model), Path(args.player_model)]
    if len({path.resolve() for path in paths}) != len(paths):
        parser.error("Entrada, saídas e modelos devem ter caminhos diferentes.")

    capture = cv2.VideoCapture(args.source)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo: {args.source}")
        fps = capture.get(cv2.CAP_PROP_FPS)
    finally:
        capture.release()
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
        print("FPS indisponível; a usar 25 FPS.")

    started = perf_counter()
    records = track_video(args.source, args.player_model)
    finish_phase("1. Tracking do vídeo completo", started)

    started = perf_counter()
    pitch = project_video(args.source, args.model, records)
    finish_phase("2. Homografia e projeção dos jogadores", started)

    started = perf_counter()
    positions_path.parent.mkdir(parents=True, exist_ok=True)
    with positions_path.open("w", encoding="utf-8") as file:
        json.dump({
            "source": args.source, "fps": fps,
            "pitch_m": {"length": pitch.length / 100, "width": pitch.width / 100},
            "frames": records,
        }, file, ensure_ascii=False, allow_nan=False)
    finish_phase("3. Guardar lista de posições", started)
    print(f"Posições guardadas em: {positions_path}")

    started = perf_counter()
    render_video(args.source, args.output, records, pitch, fps)
    finish_phase("4. Desenhar e guardar vídeo", started)
    calibrated = sum(record["homography"] is not None for record in records)
    print(f"Vídeo guardado em: {args.output}")
    print(f"Homografia calculada em {calibrated}/{len(records)} frames.")


if __name__ == "__main__":
    main()
