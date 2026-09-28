import cv2


def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Pixel coordinates X (u) = {x}, Y (V) = {y}")

cap = cv2.VideoCapture(1)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


if not cap.isOpened():
    print("Camera could not be opened")
    exit()

cv2.namedWindow("window_name")
cv2.setMouseCallback("Camera Feed", click_event)

"""
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"FPS: {fps}")
"""

while True:
    ret, frame= cap.read()
    if not ret: 
        print("Could not read frame")
        break 
    cv2.imshow('Window_name', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

