from ultralytics import YOLO
import json
import os


# 字典
dict_name = {0: 'Car', 
            1: 'LightTruck', 
            2: 'LargeBus', 
            3: 'van', 
            4: 'Truck', 
            5: 'Pedestrian', 
            6: 'TricycleClosed', 
            7: 'Pickup', 
            8: 'Motorcycle', 
            9: 'HeavyTruck', 
            10: 'MotorCyclist', 
            11: 'EngineTruck', 
            12: 'Machineshop', 
            13: 'BiCyclist', 
            14: 'TricycleOpenMotor', 
            15: 'Bike', 
            16: 'TricycleOpenHuman', 
            17: 'OtherCar', 
            18: 'MediumBus', 
            19: 'PersonSitting', 
            20: 'CampusBus', 
            21: 'MMcar'}
    
    

# box的坐标转换
def xywh2xyxy(x, y, w, h):
    width = 1280
    height = 720
    xcenter = x * width
    ycenter = y * height
    x1 = round(xcenter - w * width / 2, 1)
    y1 = round(ycenter - h * height / 2, 1)
    x2 = round(xcenter + w * width / 2, 1)
    y2 = round(ycenter + h * height / 2, 1)
    return x1, y1, x2, y2
    

h = 720
w = 1280

outs = {}
outs['annotations'] = []

def txt2json(txt_path, json_path):
    txt_list = os.listdir(txt_path)
    for txt in txt_list:
        file_path = os.path.join(txt_path, txt)
        with open(file=file_path, mode='r', encoding='utf-8') as f:
            for line in f:
                if line:
                    line = line.strip().split(' ')
                    one = {}
                    txt_name = txt.split('.')[0] + '.jpg'
                    one['filename'] = 'test_images\\' + txt_name
                    one['conf'] = round(float(line[5]), 2)                
                    id = int(line[0])
                    one['label'] = dict_name[id]
                    x = float(line[1])
                    y = float(line[2])
                    w = float(line[3])
                    h = float(line[4])
                    xmin, ymin, xmax, ymax = xywh2xyxy(x, y, w, h)
                    one['box'] = {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}
                    outs['annotations'].append(one)

  
    
if __name__ == '__main__':

    # 请修改为最新路径
    best_weight = "./runs/detect/train/weights/best.pt"
    test_images = "./datasets/cars/images/test/"

    # 模型预测
    model = YOLO(model=best_weight)
    model.predict(source=test_images, save_txt=True, save_conf=True)

    # 预测结果输出路径
    txt_path = "./runs/detect/predict/labels"
    json_path = "./output/output_dets.json"

    # 把结果转成json格式
    txt2json(txt_path, json_path)

    # 输出json文件，并提交官网查看分数
    # https://www.datafountain.cn/competitions/552/submits?view=submit
    with open(json_path, 'w') as f:
        json.dump(outs, f)  


