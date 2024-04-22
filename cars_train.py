from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO(model="yolov8n.yaml")
    # model = YOLO(model="./runs/detect/train/weights/best.pt")

    results = model.train(data='cars_v3.yaml', epochs=300, imgsz=640, batch=64)
