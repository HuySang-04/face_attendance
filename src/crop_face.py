from utils import *

save_dir = '../outputs/crop_face_dataset'
os.makedirs(save_dir, exist_ok=True)

name = input("Enter your name: ").strip()
person_dir = os.path.join(save_dir, name)
os.makedirs(person_dir, exist_ok=True)

frame_count = 0
num_skip_frame = 1
count = 5

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(
    model_selection=0, min_detection_confidence=0.7)

print("\nPress [ENTER] to capture, [ESC] to exit.\n")

cap = init_camera()
prev_frame_time = 0
new_frame_time = 0

while cap.isOpened() and count:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    ori_frame = frame.copy()

    if not ret:
        print('Cannot read camera')
        break

    if frame_count % num_skip_frame == 0:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_detection.process(frame_rgb)
        frame_count = 0
    else:
        faces = []
        start_time = time.time()
        frame_count += 1
        continue

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

    cv2.putText(frame, f"Remaining: {count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "ENTER: Capture | ESC: Exit", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    new_frame_time = time.time()
    fps_manual = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame,  f"FPS: {int(fps_manual)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Face Capture", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == 13:
        if faces:
            aligned = align_face(ori_frame, faces.detections[0], desired_size=(160, 160))
            filename = os.path.join(person_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")
            cv2.imwrite(filename, aligned)
            print(f"Saved: {filename}")
            count -= 1
        else:
            print("No face detected. Try again.")

cap.release()
cv2.destroyAllWindows()