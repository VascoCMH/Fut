from utils import read_video, save_video, camera_moved
from trackers import Tracker
from homografia import Homography
from homografia.minimap import PitchMinimap
from team_assigner import TeamAssigner
from reid import ReIdentifier, summarize
from ball import BallTracker, PossessionAssigner, draw_possession
def main():
    #read the video
    video_path="input_videos/inputvideo.mp4"

    fps, frames=read_video(video_path)
    #do something with the frames
    tracker = Tracker("modelo/player.pt", fps=fps)
    tracks = tracker.get_tracks(frames)

    ##BOLA (Remove falsos positivos; usa só píxeis, por isso vem antes da homografia)###
    ball_tracker = BallTracker()
    ball_tracker.clean(tracks["ball"])


    ##HOMOGRAPHY (Recebe key points da matriz e tranforma pixeis em coordenadas)###
    homography = Homography(model_path="modelo/pitch.pt")
    calibrated_frames = []
    matrices = []
    previous_frame = None
    matrix = None
    for frame_index, frame in enumerate(frames):
        if matrix is None or camera_moved(previous_frame, frame):
            try:
                matrix = homography.get_matrix(frame)
                previous_frame = frame
            except ValueError as error:
                print(f"Frame {frame_index + 1}: {error}")
                matrix = None
        calibrated_frames.append(matrix is not None)
        matrices.append(matrix)

        frame_tracks = {
            category: category_tracks[frame_index]
            for category, category_tracks in tracks.items()
        }
        homography.update_tracks(matrix, frame_tracks)

    ##EQUIPAS (Separa equipas, guarda-redes e árbitros pela cor da camisola)###
    team_assigner = TeamAssigner()
    team_assigner.assign(frames, tracks, homography.pitch)

    ##REID (Une tracks da mesma pessoa que desapareceu e voltou)###
    reidentifier = ReIdentifier(fps=fps)
    reidentifier.assign(tracks)
    print("\n".join(summarize(reidentifier)))

    ##BOLA (Preenche lacunas em linha reta, em metros, e calcula a posse)###
    ball_tracker.interpolate(tracks["ball"], matrices)
    print(f"Bola: {ball_tracker.removed} falsos positivos removidos, "
          f"{ball_tracker.recovered} recuperados, "
          f"{ball_tracker.interpolated} posições interpoladas")
    possession = PossessionAssigner()
    possession.assign(tracks)
    stats = possession.statistics()
    print(f"Posse: equipa 0 {stats[0]:.0f}% | equipa 1 {stats[1]:.0f}% | "
          f"disputada {stats['disputed']:.0f}%")

    drawn_frames = tracker.draw_annotations(frames, tracks)

    ##MINIMAP (Desenha a posição dos jogadores no campo)###
    minimap = PitchMinimap(homography.pitch)
    for frame_index, frame in enumerate(drawn_frames):
        people = [
            (person.get("global_id", track_id), person["position_m"],
             person.get("team_color", (0, 220, 255)))
            for category in ("players", "referees")
            for track_id, person in tracks[category][frame_index].items()
            if person["position_m"] is not None
        ]
        minimap.overlay(frame, people, calibrated_frames[frame_index])
        draw_possession(frame, possession.possession[:frame_index + 1],
                        team_assigner.team_colors)

    #save the video
    output_path="output_videos/outputvideo.mp4"
    save_video(drawn_frames, output_path, fps)

if __name__ == "__main__":
    main()
