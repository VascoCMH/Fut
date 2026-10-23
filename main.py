from utils import read_video, save_video
from trackers import Tracker
def main():
    #read the video
    video_path="input_videos/inputvideo.mp4"

    fps, frames=read_video(video_path)
    #do something with the frames
    tracker = Tracker("modelo/player.pt")
    tracks = tracker.get_tracks(frames)
    drawn_frames = tracker.draw_annotations(frames, tracks)

    #save the video
    output_path="output_videos/outputvideo.mp4"
    save_video(drawn_frames, output_path, fps)

if __name__ == "__main__":
    main()