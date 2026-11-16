from utils import read_video, save_video, camera_moved
from trackers import Tracker
from homografia import Homography
from homografia.minimap import PitchMinimap
from team_assigner import TeamAssigner
from reid import ReIdentifier, summarize
def main():
    #read the video
    video_path="input_videos/inputvideo.mp4"

    fps, frames=read_video(video_path)
    #do something with the frames
    tracker = Tracker("modelo/player.pt", fps=fps)
    tracks = tracker.get_tracks(frames)


    ##HOMOGRAPHY (Recebe key points da matriz e tranforma pixeis em coordenadas)###
    homography = Homography(model_path="modelo/pitch.pt")
    calibrated_frames = []
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

    #save the video
    output_path="output_videos/outputvideo.mp4"
    save_video(drawn_frames, output_path, fps)

if __name__ == "__main__":
    main()
