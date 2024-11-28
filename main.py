import cv2
import serial
import time
import threading
from queue import Queue
from ultralytics import YOLO


def calculate_distance(known_height, focal_length, bbox_height):
    distance = (known_height * focal_length) / bbox_height
    return distance


def calculate_move(x, distant):
    move_value = round((x - 340) / 12)
    if move_value > 30:
        move_value = 30
    elif move_value < -30:
        move_value = -30
    return move_value


def send_data_to_arduino(arduino, move_value):
    signal = f"moveto_{move_value}"
    arduino.write(f"{signal}*".encode())
    print(f"Sent to Arduino: {signal}")


def arduino_communication(arduino, data_queue):
    while True:
        if not data_queue.empty():
            data = data_queue.get()
            send_data_to_arduino(arduino, data)
        time.sleep(0.1)  # Send data every seconds


def main():
    # Define constants
    KNOWN_HEIGHT = 1.6 * 2  # Approximate height of a person in meters
    BAUD_RATE = 9600

    # Initialize YOLO model
    model = YOLO("./weights/yolo11n.pt")
    cap = cv2.VideoCapture(0)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Queue for sending data to Arduino
    data_queue = Queue(maxsize=1)

    try:
        # Uncomment and configure the correct port to connect Arduino
        # arduino_port = "/dev/ttyUSB0"
        # arduino = serial.Serial(arduino_port, BAUD_RATE, timeout=1)
        # time.sleep(2)  # Allow connection to stabilize
        # print(f"Connected to Arduino on port {arduino_port}")
        pass
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return

    # Start a thread for Arduino communication
    # arduino_thread = threading.Thread(target=arduino_communication, args=(arduino, data_queue))
    # arduino_thread.daemon = True
    # arduino_thread.start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model.track(
            frame, stream=True, persist=True, classes=[0]
        )  # Detect persons (class 0)

        annotated_frame = frame.copy()  # Copy the frame to draw annotations
        person_count = 0
        data_to_send = "0"

        for result in results:  # Iterate through the generator
            if result.boxes:
                for box in result.boxes:
                    x, y, w, h = box[0].xywh.cpu()[0]
                    x, y, w, h = int(x), int(y), int(w), int(h)

                    distance = calculate_distance(KNOWN_HEIGHT, frame_height, h)
                    move_value = calculate_move(x, distance)
                    data_to_send = move_value

                    person_count += 1

                    # Draw bounding box and label
                    cv2.rectangle(
                        annotated_frame,
                        (int(x - w / 2), int(y - h / 2)),
                        (int(x + w / 2), int(y + h / 2)),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        annotated_frame,
                        f"Person {person_count}",
                        (int(x - w / 2), int(y - h / 2 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )
                    coords_text = f"({int(x)}, {int(y)})"
                    cv2.putText(
                        annotated_frame,
                        coords_text,
                        (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2,
                    )

        # Update the data queue for Arduino communication
        if data_queue.empty():
            data_queue.put("0" if person_count == 0 else data_to_send)

        # Display the frame
        cv2.imshow("YOLO Person Detection", annotated_frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    # arduino.close()
    cv2.destroyAllWindows()
    print("Program terminated.")


if __name__ == "__main__":
    main()
