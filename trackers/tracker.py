from ultralytics import YOLO
import supervision as sv

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_track(self, frames):
        results=[]
        batch_size = 32  # É a quantidade de frames que se procesa de cada vez
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_results = self.model.predict(batch_frames,conf=0.1)
            results += batch_results
        return results

    def get_tracks(self, frames):
        detections = self.detect_track(frames)

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names