import mediapipe as mp
import cv2

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(
    model_selection=0, min_detection_confidence=0.7)

def detect_face_media(frame, count):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detection.process(frame_rgb)

    if faces.detections:
        face = faces.detections[0]
        height, width, _ = frame.shape

        bbox = face.location_data.relative_bounding_box
        x = int(bbox.xmin * width)
        y = int(bbox.ymin * height)
        w = int(bbox.width * width)
        h = int(bbox.height * height)

        conf = face.score[0]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{conf:.2f}", (x + 3, y + h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # if count >0:
        #     cv2.putText(frame, f'remaining: {count}', (x, y - 20),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame, face
    return None, None
