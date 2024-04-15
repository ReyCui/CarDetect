from ultralytics import YOLO

if __name__ == '__main__':

    # model = YOLO(model="yolov8n.yaml")
    model = YOLO(model="./runs/detect/train/weights/best.pt")

    results = model.train(data='cars.yaml', epochs=200, imgsz=512, batch=64)