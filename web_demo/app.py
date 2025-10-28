from flask import Flask, render_template, request, jsonify, Response
import time
from crop_face import detect_face_media
import threading

from utils import *
from train import update_embeddings
from diemdanh import AttendanceSystem

app = Flask(__name__)

cap = None
ori_frame = None
box = None
dict_images = {}
current_name = None
count = 5
num_skip_frame=2
frame_count=0

attendance_system = None
is_running = False
current_status = "Chưa khởi động"
attendance_log = []
all_students = []

@app.route('/')
def home():
    return render_template('home.html')

#==================================================================
#                        CROP FACE ROUTES
#==================================================================
def gen_frames():
    global cap, ori_frame, box, count, frame_count, num_skip_frame
    while cap and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_count % num_skip_frame == 0:
            frame_count = 0
            ori_frame = frame.copy()
            frame, box = detect_face_media(frame, count)

            if frame is None and box is None:
                continue

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)
        frame_count += 1

@app.route('/crop_face')
def crop_face_page():
    return render_template('crop_face.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global cap, count
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(1)
        count = 5
        return jsonify({"status": "started"})
    return jsonify({"status": "already running"})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global cap, dict_images, current_name, count
    if cap and cap.isOpened():
        cap.release()
    cap = None
    dict_images = {}
    current_name = None
    count = 5
    return jsonify({"status": "stopped"})

@app.route('/video_feed')
def video_feed():
    if cap and cap.isOpened():
        return Response(gen_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response("Camera not available", status=500)

@app.route('/crop_face', methods=['POST'])
def crop_face():
    global ori_frame, box, current_name, dict_images, count

    try:
        data = request.get_json()
        name = data.get('name', '').strip()

        if not name:
            return jsonify({"status": "error", "message": "Tên không được để trống"}), 400
        if current_name != name:
            current_name = name
            dict_images = {}
            count = 5

        if ori_frame is not None and box is not None and count > 0:
            aligned_face = align_face(ori_frame, box, (160, 160))
            save_path = f'./data/{name}/{count}.jpg'

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            dict_images[save_path] = aligned_face
            count -= 1

            return jsonify({
                "status": "crop success",
                "count": count,
                "name": name,
                "message": f"Đã crop khuôn mặt. Còn lại {count} lần crop."
            })
        return jsonify({"status": "error", "message": "Không thể crop khuôn mặt"}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/update', methods=['POST'])
def update():
    global current_name, dict_images, count
    try:
        if not current_name:
            return jsonify({"status": "error", "message": "Chưa có dữ liệu để cập nhật"}), 400
        if not dict_images:
            return jsonify({"status": "error", "message": "Không có ảnh nào để lưu"}), 400

        for key, val in dict_images.items():
            cv2.imwrite(key, val)
        update_success = update_embeddings(current_name)

        dict_images = {}
        current_name = None
        count = 5

        if update_success:
            return jsonify({
                "status": "success",
                "message": "Đã cập nhật embeddings thành công"
            })
        else:
            return jsonify({
                "status": "error",
                "message": "Không thể cập nhật embeddings"
            }), 500

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


#==================================================================
#                        ATTENDANCE ROUTES
#==================================================================
@app.route('/attendance')
def attendance_page():
    return render_template('attendance.html')

@app.route('/start_attendance', methods=['POST'])
def start_attendance():
    global attendance_system, is_running, current_status, all_students

    if is_running:
        return jsonify({"status": "error", "message": "Hệ thống đang chạy"})
    try:
        data = request.get_json()
        duration = int(data.get('duration', 30))
        is_start = bool(data.get('is_start', False))

        is_end = bool(data.get('is_end', False))
        attendance_system = AttendanceSystem()
        is_running = True

        if is_start and is_end:
            current_status = "Đang điểm danh - Cuối ca"
        elif is_start:
            current_status = "Đang điểm danh - Đầu ca"
        elif is_end:
            current_status = "Đang điểm danh - Cuối ca"
        else:
            current_status = "Đang điểm danh - Đầu ca"

        thread = threading.Thread(
            target=attendance_system.start_attendance,
            args=(duration, is_start, is_end)
        )
        thread.daemon = True
        thread.start()

        time.sleep(2)
        if attendance_system and hasattr(attendance_system, 'all_students'):
            all_students = attendance_system.all_students.copy()

        return jsonify({
            "status": "success",
            "message": f"Bắt đầu điểm danh trong {duration} phút",
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/stop_attendance', methods=['POST'])
def stop_attendance():
    global attendance_system, is_running, current_status

    if attendance_system and is_running:
        attendance_system.stop()
        is_running = False
        current_status = "Đã dừng"
        return jsonify({"status": "success", "message": "Đã dừng điểm danh"})
    return jsonify({"status": "error", "message": "Hệ thống không đang chạy"})

def generate_attendance_frames():
    while True:
        if (attendance_system and attendance_system.is_running and
                attendance_system.current_frame is not None):
            frame = attendance_system.current_frame
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    time.sleep(0.1)
            except Exception as e:
                print(f"Lỗi encode frame: {e}")
                time.sleep(0.1)
        else:
            time.sleep(0.1)

@app.route('/attendance_video_feed')
def attendance_video_feed():
    return Response(generate_attendance_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_status')
def get_status():
    global attendance_system, is_running, current_status, attendance_log

    status_data = {
        "is_running": is_running,
        "status": current_status,
        "detected_count": attendance_system.detected_count if attendance_system else 0,
        "fps": attendance_system.fps if attendance_system else 0,
        "attendance_count": len(attendance_system.attendance_dict) if attendance_system else 0,
        "log": attendance_log[-10:]
    }
    return jsonify(status_data)

@app.route('/get_all_students')
def get_all_students():
    global attendance_system, all_students

    students_data = []
    try:
        if attendance_system and hasattr(attendance_system, 'all_students'):
            if attendance_system.all_students:
                all_students = attendance_system.all_students.copy()
            for student_name in all_students:
                attended = student_name in attendance_system.attendance_dict
                attendance_time = attendance_system.attendance_dict.get(student_name, "")
                students_data.append({
                    "name": student_name,
                    "attended": attended,
                    "time": attendance_time
                })
        else:
            if os.path.exists(LIST_NAME):
                names_lists = np.load(LIST_NAME, allow_pickle=True)
                unique_names = list(np.unique(names_lists))
                all_students = unique_names

                for student_name in unique_names:
                    students_data.append({
                        "name": student_name,
                        "attended": False,
                        "time": ""
                    })
            else:
                students_data = []

    except Exception as e:
        print(f"Lỗi khi lấy danh sách sinh viên: {e}")
        students_data = []

    return jsonify(students_data)

@app.route('/get_students_list')
def get_students_list():
    global all_students

    try:
        if not all_students and os.path.exists(LIST_NAME):
            names_lists = np.load(LIST_NAME, allow_pickle=True)
            all_students = list(np.unique(names_lists))

        return jsonify({
            "students": all_students,
            "count": len(all_students)
        })
    except Exception as e:
        return jsonify({"error": str(e), "students": [], "count": 0})

@app.route('/reset_attendance')
def reset_attendance():
    global attendance_system, is_running, current_status
    if attendance_system:
        attendance_system.stop()
    attendance_system = None
    is_running = False
    current_status = "Đã reset"
    return jsonify({"status": "success", "message": "Đã reset hệ thống điểm danh"})

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)

    try:
        if os.path.exists(LIST_NAME):
            names_lists = np.load(LIST_NAME, allow_pickle=True)
            all_students = list(np.unique(names_lists))
            print(f"Đã load {len(all_students)} sinh viên từ database")
        else:
            print("Chưa có dữ liệu sinh viên. Vui lòng đăng ký khuôn mặt trước.")
    except Exception as e:
        print(f"Lỗi khi load danh sách sinh viên: {e}")

    app.run(debug=True, host='0.0.0.0', port=5000)