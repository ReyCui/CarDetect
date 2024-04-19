from ultralytics import YOLO

if __name__ == '__main__':

    # 从配置文件，从零初始化一个对象
    model = YOLO(model="test_yolov8s.yaml")

    # Train the model
    # results = model.train(data='cars.yaml', epochs=10, imgsz=1280, batch=8)
    results = model.train(data='cars_v3.yaml',
                          epochs=300,
                          imgsz=640,
                          batch=16,
                          cos_lr=True,
                          lr0=0.0001,
                          lrf=0.001,
                          dropout=0.6,
                          patience=50,
                          multi_scale=True,
                          mask_ratio=10)