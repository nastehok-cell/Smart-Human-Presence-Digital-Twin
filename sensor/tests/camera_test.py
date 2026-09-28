import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(1)
cap.set(3, 1920)
cap.set(4, 1080)

fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"FPS: {fps}")

while True:
    ret, img = cap.read()
    if not ret:
        print("Ei saatu kuvaa kamerasta")
        break

    # Näytetään alkuperäinen kuva
    results = model(img)

    annotated_frame = results[0].plot()

    #Näytetään YOLO tulokset
    cv2.imshow('YOLO WebCam', annotated_frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()