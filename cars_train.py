from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model="yolov8n.yaml")

    results = model.train(data='cars.yaml', epochs=2, imgsz=128, batch=8)