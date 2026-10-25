import cv2  
  
points = []

#Käsitellään hiiren klikkaukset
def click_event(event, x, y, flags, param): 
    global annotated_frame
 
    if event == cv2.EVENT_LBUTTONDOWN:  
        font = cv2.FONT_HERSHEY_SIMPLEX
        print(f'({x},{y})')
        points.append((x, y))
        
  
from ultralytics import YOLO 

#Ladataan YOLO-malli
model = YOLO("yolov8n.pt") 
 
cap = cv2.VideoCapture(0) 

#Resoluutio
cap.set(3, 1280) 
cap.set(4, 720) 
 
 
if not cap.isOpened():  
    print("Camera could not be opened")  
    exit()  
  
window_name = "Camera Feed"  
  
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  
cv2.setMouseCallback(window_name, click_event)  
  
while True: 
    ret, frame = cap.read() 
    if not ret: 
        print("Could not read frame") 
        break 

    
    # Näytetään alkuperäinen kuva
    results = model(frame) 
    
    #Piirretään käyttäjän klikatut pisteet
    annotated_frame = results[0].plot() 
    for (x, y) in points: 
         cv2.circle(annotated_frame, (x,y), 3, (0,255,255), -1)
         cv2.putText(annotated_frame, f'({x},{y})', (x,y),
                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    #Näytetään YOLO tulokset
    cv2.imshow(window_name, annotated_frame) 
     
    if cv2.waitKey(1) == ord('q'):
            break
 
cap.release() 
cv2.destroyAllWindows()