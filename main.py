from utils import read_video, save_video

def main():
    #read the video
    video_path="input_videos/inputvideo.mp4"

    fps, frames=read_video(video_path)
    #do something with the frames

    #save the video
    output_path="output_videos/outputvideo.mp4"
    save_video(frames, output_path, fps)

if __name__ == "__main__":
    main()