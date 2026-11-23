"""Fluxo principal em streaming: o vídeo nunca está todo em memória.

Passagem 1 (análise): lê o vídeo em lotes, faz deteção e tracking,
homografia e recolha das cores das camisolas; no fim, com as tracks (dados
pequenos), corre equipas, ReID, bola e posse.
Passagem 2 (desenho): volta a ler o vídeo, desenha cada frame e escreve-o
logo no ficheiro de saída.
"""

from collections import Counter

import cv2

from utils import Timings, camera_moved, iter_batches, iter_frames, video_info
from trackers import Tracker
from homografia import Homography
from homografia.minimap import PitchMinimap
from team_assigner import TeamAssigner
from reid import ReIdentifier, summarize
from ball import BallTracker, PossessionAssigner, draw_possession
from positions import HomographyStabilizer, PositionSmoother

VIDEO_PATH = "input_videos/inputvideo.mp4"
OUTPUT_PATH = "output_videos/outputvideo.mp4"
BATCH_SIZE = 32
timings = Timings()


def analyse(video_path):
    """Passagem 1: devolve tracks, matrizes de homografia e o TeamAssigner."""
    fps, total, width, height = video_info(video_path)
    tracker = Tracker("modelo/player.pt", fps=fps)
    homography = Homography(model_path="modelo/pitch.pt")
    team_assigner = TeamAssigner()

    tracks = {"players": [], "referees": [], "ball": []}
    matrices = []
    reference_frame, matrix = None, None
    frame_index = 0
    for batch in iter_batches(video_path, BATCH_SIZE):
        ##TRACKING (o mesmo tracker mantém o estado entre lotes)###
        with timings.measure("Deteção + tracking (YOLO)"):
            batch_tracks = tracker.get_tracks(batch)
        for category in tracks:
            tracks[category].extend(batch_tracks[category])

        for frame in batch:
            ##HOMOGRAPHY (recalibra só quando a câmara se mexe)###
            with timings.measure("Movimento da câmara"):
                moved = matrix is None or camera_moved(reference_frame, frame)
            if moved:
                try:
                    with timings.measure("Homografia (modelo do campo)"):
                        matrix = homography.get_matrix(frame)
                    reference_frame = frame
                except ValueError as error:
                    print(f"Frame {frame_index + 1}: {error}")
                    matrix = None
            matrices.append(matrix)
            homography.update_tracks(matrix, {
                category: category_tracks[frame_index]
                for category, category_tracks in tracks.items()
            })

            ##EQUIPAS (recolhe as cores das camisolas deste frame)###
            with timings.measure("Cores das camisolas"):
                team_assigner.add_frame(frame_index, frame, tracks["players"][frame_index])
            frame_index += 1
        print(f"Análise: {frame_index}/{total} frames", flush=True)
    print(f"Deteção a correr em: {tracker.model.device}")
    return fps, (width, height), tracker, homography, team_assigner, tracks, matrices


def main():
    fps, frame_size, tracker, homography, team_assigner, tracks, matrices = analyse(VIDEO_PATH)
    timings.start("Equipas, ReID, bola, posições")

    ##ESTABILIZAÇÃO (corrige o erro da homografia comum a todos os jogadores)###
    stabilizer = HomographyStabilizer(frame_size=frame_size)
    matrices = stabilizer.stabilize(tracks, matrices)
    corrected = sum(c > 0.5 for c in stabilizer.corrections)
    print(f"Homografia: {corrected} frames corrigidos em mais de 0,5 m")

    ##BOLA (remove falsos positivos)###
    ball_tracker = BallTracker()
    ball_tracker.clean(tracks["ball"])

    # Recalcular todas as posições em metros com as matrizes corrigidas.
    for frame_index, matrix in enumerate(matrices):
        homography.update_tracks(matrix, {
            category: category_tracks[frame_index]
            for category, category_tracks in tracks.items()
        })

    ##EQUIPAS (separa equipas, guarda-redes e árbitros pela cor da camisola)###
    team_assigner.assign(None, tracks, homography.pitch)

    ##REID (une tracks da mesma pessoa que desapareceu e voltou)###
    reidentifier = ReIdentifier(fps=fps)
    reidentifier.assign(tracks)
    print("\n".join(summarize(reidentifier)))

    ##SUAVIZAÇÃO (remove saltos, suaviza e estima as ausências em linha reta)###
    smoother = PositionSmoother(fps=fps, frame_size=frame_size)
    smoother.smooth(tracks)
    print(f"Posições: {smoother.outliers} saltos e {smoother.border} caixas "
          f"cortadas pela borda descartados; {smoother.estimated} posições "
          f"estimadas em ausências")

    ##BOLA (preenche lacunas em linha reta, em metros, e calcula a posse)###
    ball_tracker.interpolate(tracks["ball"], matrices)
    print(f"Bola: {ball_tracker.removed} falsos positivos removidos, "
          f"{ball_tracker.recovered} recuperados, "
          f"{ball_tracker.interpolated} posições interpoladas")
    possession = PossessionAssigner()
    possession.assign(tracks)
    stats = possession.statistics()
    print(f"Posse: equipa 0 {stats[0]:.0f}% | equipa 1 {stats[1]:.0f}% | "
          f"disputada {stats['disputed']:.0f}%")

    timings.stop("Equipas, ReID, bola, posições")

    ##DESENHO (passagem 2: lê, desenha e escreve frame a frame)###
    timings.start("Desenho e escrita do vídeo")
    _, total, width, height = video_info(VIDEO_PATH)
    writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    minimap = PitchMinimap(homography.pitch)
    counts = Counter()
    for frame_index, frame in enumerate(iter_frames(VIDEO_PATH)):
        if frame_index >= len(tracks["players"]):
            break
        tracker.draw_frame(frame, frame_index, tracks)
        people = [
            (person.get("global_id", track_id),
             person.get("position_smooth") or person["position_m"],
             person.get("team_color", (0, 220, 255)), False)
            for category in ("players", "referees")
            for track_id, person in tracks[category][frame_index].items()
            if person["position_m"] is not None
        ]
        people += [
            (gid, person["position_smooth"], person.get("team_color"), True)
            for gid, person in tracks["estimated"][frame_index].items()
        ]
        minimap.overlay(frame, people, matrices[frame_index] is not None)
        current = possession.possession[frame_index]
        if current is not None:
            counts[current] += 1
        draw_possession(frame, counts, current, team_assigner.team_colors)
        writer.write(frame)
        if (frame_index + 1) % 500 == 0:
            print(f"Desenho: {frame_index + 1}/{total} frames", flush=True)
    writer.release()
    timings.stop("Desenho e escrita do vídeo")
    print(f"Vídeo guardado em {OUTPUT_PATH}")
    print(timings.report(video_seconds=len(tracks["players"]) / fps if fps else None))


if __name__ == "__main__":
    main()
