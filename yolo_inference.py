from ultralytics import YOLO

model=YOLO("modelo/pitch.pt")  # Load a pretrained YOLOv11m model (Medium model)

result=model.predict("input_videos/inputvideo.mp4", save=True)
print(result)  # Print the results of the prediction