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