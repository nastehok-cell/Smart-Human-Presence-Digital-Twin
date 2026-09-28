import cv2

stream = cv2.VideoCapture(1)

if not stream.isOpened():
    print("No stream: (")
    exit()

fps = stream.get(cv2.CAP_PROPS_FPS)
width =stream.get(3)
height = stream.get(4)

output = cv2.VideoWriter("assets/4_stream.mp4",
        cv2.VideoWriter_fourcc("m", "p", "4", "v"),
        fps=fps, frameSize=(width, height))


stream.release()
cv2.destroyAllWindows()