from ultralytics import YOLO


# YOLOv8 mallin lataaminen
model = YOLO("yolov8n.pt")


# # Suorita YOLO testikuvalla
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    save=True,
)


print("YOLO test completed successfully.") 