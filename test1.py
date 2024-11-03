from flask import Flask, render_template, Response, request
import serial.tools.list_ports
import cv2
import threading
import numpy as np

app = Flask(__name__)

# Global serial instance
serial_inst = None

# รายชื่อหมวดหมู่ทั้งหมด เรียงตามลำดับ
CLASSES = ["BACKGROUND", "AEROPLANE", "BICYCLE", "BIRD", "BOAT",
           "BOTTLE", "BUS", "CAR", "CAT", "CHAIR", "COW", "DININGTABLE",
           "DOG", "HORSE", "MOTORBIKE", "PERSON", "POTTEDPLANT", "SHEEP",
           "SOFA", "TRAIN", "TVMONITOR"]
# สีตัวกรอบที่วาด random ใหม่ทุกครั้ง
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# โหลด model จากแฟ้ม
net = cv2.dnn.readNetFromCaffe("./MobileNetSSD/MobileNetSSD.prototxt", "./MobileNetSSD/MobileNetSSD.caffemodel")

# Function to list all available ports
def list_ports():
    ports = serial.tools.list_ports.comports()
    ports_list = []
    for one in ports:
        ports_list.append(str(one))
    return ports_list

# Function to select the correct COM port
def select_port():
    ports_list = list_ports()
    if not ports_list:
        print("No ports found.")
        return None

    print("Available Ports:")
    for i, port in enumerate(ports_list):
        print(f"{i + 1}: {port}")

    selected_port = input("Select COM Port #: ")
    try:
        index = int(selected_port) - 1
        if 0 <= index < len(ports_list):
            return ports_list[index].split(' ')[0]
        else:
            print("Invalid selection.")
            return None
    except ValueError:
        print("Invalid input.")
        return None

# Function to handle the camera display with object detection
def handle_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break

        (h, w) = frame.shape[:2]
        # ทำ preprocessing
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        # Feed เข้า model พร้อมได้ผลลัพธ์ทั้งหมดเก็บมาในตัวแปร detections
        detections = net.forward()

        # ตัวแปรสำหรับนับจำนวนบุคคลที่ตรวจจับได้
        person_count = 0

        for i in np.arange(0, detections.shape[2]):
            percent = detections[0, 0, i, 2]
            # กรองเอาเฉพาะค่า percent ที่สูงกว่า 0.2 เพิ่มลดได้ตามต้องการ
            if percent > 0.3:
                class_index = int(detections[0, 0, i, 1])
                # ตรวจสอบให้ตรวจจับเฉพาะ "PERSON"
                if CLASSES[class_index] == "PERSON":
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # ส่วนตกแต่งสามารถลองแก้กันได้ วาดกรอบและชื่อ
                    label = "{} [{:.2f}%]".format(CLASSES[class_index], percent * 100)
                    # เปลี่ยนสีและความหนาของกรอบเพื่อให้เด่นขึ้น
                    color = (0, 255, 0)  # สีเขียวเข้ม
                    thickness = 4        # ความหนาของกรอบ
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, thickness)
                    cv2.rectangle(frame, (startX - 1, startY - 30), (endX + 1, startY), color, cv2.FILLED)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX + 20, y + 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

                    # เพิ่มจำนวนบุคคลที่ตรวจจับได้
                    person_count += 1

        # แสดงจำนวนบุคคลที่ตรวจจับได้ที่มุมจอซ้ายบน
        count_label = "People Detected: {}".format(person_count)
        cv2.putText(frame, count_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(handle_camera(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/send_command', methods=['POST'])
def send_command():
    global serial_inst
    command = request.form.get('command')
    if serial_inst and serial_inst.is_open:
        serial_inst.write(command.encode('utf-8'))
        if command.lower() == 'exit':
            serial_inst.close()
    return '', 204

def main():
    global serial_inst
    selected_port = select_port()
    if not selected_port:
        return

    serial_inst = serial.Serial()
    serial_inst.baudrate = 9600
    serial_inst.port = selected_port
    serial_inst.open()

    # Run the Flask app with access from all IPs in the local network
    app.run(host='0.0.0.0', port=5000, threaded=True)  # Enable threading

if __name__ == "__main__":
    main()