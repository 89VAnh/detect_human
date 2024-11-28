import cv2

def list_available_cameras(max_index=10):
    available_cameras = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Camera found at index {index}")
            available_cameras.append(index)
            cap.release()
    if not available_cameras:
        print("No available cameras found.")
    return available_cameras

available_cameras = list_available_cameras()
print("Available camera indices:", available_cameras)
