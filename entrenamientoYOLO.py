from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(
    data="C:/Users/Alex/Downloads/archive/data.yaml",
    epochs=20,
    imgsz=640,
    batch=8,
    device="cpu",  
    workers=0,
    single_cls=True,
    name="car_plate_detection"
)