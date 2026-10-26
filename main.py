from utils import read_video, save_video, camera_moved
from trackers import Tracker
from homografia import Homography
from homografia.minimap import PitchMinimap
def main():
    #read the video
    video_path="input_videos/inputvideo.mp4"

    fps, frames=read_video(video_path)
    #do something with the frames
    tracker = Tracker("modelo/player.pt")
    tracks = tracker.get_tracks(frames)


    ##HOMOGRAPHY (Recebe key points da matriz e tranforma pixeis em coordenadas)###
    homography = Homography(model_path="modelo/pitch.pt")
    calibrated_frames = []
    previous_frame = None
    for frame_index, frame in enumerate(frames):
        if camera_moved(previous_frame, frame):
            try:
                matrix = homography.get_matrix(frame)
            except ValueError as error:
                print(f"Frame {frame_index + 1}: {error}")
                matrix = None
            calibrated_frames.append(matrix is not None)

        frame_tracks = {
            category: category_tracks[frame_index]
            for category, category_tracks in tracks.items()
        }
        homography.update_tracks(matrix, frame_tracks)

    drawn_frames = tracker.draw_annotations(frames, tracks)

    ##MINIMAP (Desenha a posição dos jogadores no campo)###
    minimap = PitchMinimap(homography.pitch)
    for frame_index, frame in enumerate(drawn_frames):
        players = [
            (track_id, player["position_m"])
            for track_id, player in tracks["players"][frame_index].items()
            if player["position_m"] is not None
        ]
        minimap.overlay(frame, players, calibrated_frames[frame_index])

    #save the video
    output_path="output_videos/outputvideo.mp4"
    save_video(drawn_frames, output_path, fps)

if __name__ == "__main__":
    main()
