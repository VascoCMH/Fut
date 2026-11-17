import cv2
from ultralytics import YOLO
import numpy as np
import supervision as sv
from utils import get_center_of_bbox, get_bbox_width

class Tracker:
    def __init__(self, model_path, fps=25, lost_track_buffer=100):
        """lost_track_buffer: nº de frames durante os quais o ByteTrack se
        lembra de uma pessoa perdida (100 frames = 4 s a 25 fps). O valor por
        omissão do ByteTrack (30) só cobre oclusões de ~1 s.
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack(
            lost_track_buffer=lost_track_buffer,
            frame_rate=int(round(fps)) if fps else 25,
        )

    def detect_track(self, frames):
        results=[]
        batch_size = 32  # É a quantidade de frames que se procesa de cada vez
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_results = self.model.predict(batch_frames,conf=0.1)
            results += batch_results
        return results


    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_annotations(self,video_frames, tracks):
        output_video_frames= []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                label = player.get("global_id", track_id)
                frame = self.draw_ellipse(frame, player["bbox"], color, label)
                if player.get("has_ball"):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            # Draw referees
            for track_id, referee in referee_dict.items():
                color = referee.get("team_color", (0, 255, 255))
                frame = self.draw_ellipse(frame, referee["bbox"], color)

            # Draw ball 
            # Bola detetada: verde; posição interpolada: verde-claro.
            for track_id, ball in ball_dict.items():
                color = (170, 255, 170) if ball.get("interpolated") else (0, 255, 0)
                frame = self.draw_traingle(frame, ball["bbox"], color)

            output_video_frames.append(frame)

        return output_video_frames

    
    def get_tracks(self, frames):
        results = self.detect_track(frames)
        detections = [sv.Detections.from_ultralytics(r) for r in results]
        names = results[0].names if results else {}
        return self.tracks_from_detections(detections, names)

    def tracks_from_detections(self, detections, cls_names):
        """Tracking a partir de deteções já calculadas (sv.Detections por frame).

        Separado da deteção para poder repetir o tracking com outros
        parâmetros sem voltar a correr o YOLO.
        """
        cls_names_inv = {v: k for k, v in cls_names.items()}
        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection_supervision in enumerate(detections):

            # Juntar jogador, guarda-redes e árbitro numa só classe "pessoa".
            # As classes do modelo ainda não são fiáveis; a distinção é feita
            # depois pela cor (TeamAssigner). A classe original fica como pista.
            person_classes = {"player", "goalkeeper", "referee"}
            yolo_classes = np.array(
                [cls_names[c] for c in detection_supervision.class_id], dtype=object
            )
            detection_supervision.data["yolo_class"] = yolo_classes
            for object_ind, class_name in enumerate(yolo_classes):
                if class_name in person_classes:
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # O NMS do YOLO é feito por classe: a mesma pessoa pode vir em duas
            # caixas (p. ex. "player" e "goalkeeper"). Depois de fundir as
            # classes, eliminar esses duplicados para não criar tracks duplas.
            if len(detection_supervision) > 0:
                detection_supervision = detection_supervision.with_nms(
                    threshold=0.7, class_agnostic=False
                )

            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {
                        "bbox": bbox,
                        "yolo_class": frame_detection[5]["yolo_class"],
                    }
            
            # Bola: guardar a deteção mais confiante e todas as candidatas
            # (pode haver falsos positivos; o BallTracker escolhe a coerente).
            candidates = [
                {"bbox": d[0].tolist(), "confidence": float(d[2])}
                for d in detection_supervision
                if d[3] == cls_names_inv['ball']
            ]
            if candidates:
                best = max(candidates, key=lambda c: c["confidence"])
                tracks["ball"][frame_num][1] = {**best, "candidates": candidates}
        return tracks