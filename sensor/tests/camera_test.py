import cv2

cap = cv2.VideoCapture(1)
cap.set(3, 1920)
cap.set(4, 1080)

fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"FPS: {fps}")

while True:
    ret, img= cap.read()
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()