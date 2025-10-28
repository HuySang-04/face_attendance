import cv2
import os
import mediapipe as mp

base_path = "./data"
image_path = os.path.join(base_path, "images/train/fake")
label_path = os.path.join(base_path, "labels/train/fake")

os.makedirs(image_path, exist_ok=True)
os.makedirs(label_path, exist_ok=True)

mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=1, min_detection_confidence=0.5)

count = 0
class_id = 0
auto_save = False

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(frame_rgb)

    img_display = frame.copy()

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)
            cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(img_display, "'s'=auto save, 't'=stop auto save, 'q'=quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Camera", img_display)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('s'):
        auto_save = True
        print("Auto save ON")
    elif key & 0xFF == ord('t'):
        auto_save = False
        print("Auto save OFF")

    if results.detections and auto_save:
        frame_resized = cv2.resize(frame, (640, 640))
        img_name = f"{count:06d}.jpg"
        cv2.imwrite(os.path.join(image_path, img_name), frame_resized)

        ih, iw, _ = frame.shape
        scale_x = 640 / iw
        scale_y = 640 / ih

        label_lines = []
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x = bboxC.xmin * iw
            y = bboxC.ymin * ih
            w = bboxC.width * iw
            h = bboxC.height * ih

            x_scaled = x * scale_x
            y_scaled = y * scale_y
            w_scaled = w * scale_x
            h_scaled = h * scale_y
            x_center = (x_scaled + w_scaled / 2) / 640
            y_center = (y_scaled + h_scaled / 2) / 640
            w_rel = w_scaled / 640
            h_rel = h_scaled / 640

            label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_rel:.6f} {h_rel:.6f}")

        label_name = f"{count:06d}.txt"
        with open(os.path.join(label_path, label_name), "w") as f:
            f.write("\n".join(label_lines) + "\n")

        print(f"Saved {img_name} and {label_name} ({len(results.detections)} faces)")
        count += 1

cap.release()
cv2.destroyAllWindows()
face_detector.close()
