from ultralytics import YOLO
import json

def result2json(results):
    outs = {}
    outs['annotations'] = []

    for result in results:
        filename = result.path[-14:].replace('test','test_images')
        json_file = json.loads(result.tojson())

        for obj in json_file:
            # print(obj)
            one = {
                'filename': filename,
                'label': obj['name'],  # 修改 'name' 为 'label'
                'conf': round(obj['confidence'], 2),  # 保留小数点后两位
                'box': {
                    'xmin': round(obj['box']['x1'], 1),  # 保留小数点后两位
                    'ymin': round(obj['box']['y1'], 1),  # 保留小数点后两位
                    'xmax': round(obj['box']['x2'], 1),  # 保留小数点后两位
                    'ymax': round(obj['box']['y2'], 1)   # 保留小数点后两位
                }
            }
            outs['annotations'].append(one)
    
    return outs
    

  
    
if __name__ == '__main__':

    # 请修改为最新路径
    # best_weight = "./runs/detect/train6/weights/best.pt"
    best_weight = "./others/back/yolov8s/best.pt"
    test_images = "./datasets/cars/images/test/"
    json_path = "./output/output_dets_8s_best309.json"

    # 模型预测
    model = YOLO(model=best_weight)
    results = model.predict(source=test_images, save_conf=True)
    # 如果想保存图片结果，可以用下面预测代码
    # results = model.predict(source=test_images, imgsz=[1280,720], conf=0.2, save=True, save_txt=True, save_conf=True)


    # 把结果转成json格式
    outs = result2json(results=results)

    # 输出json文件，并提交官网查看分数
    # https://www.datafountain.cn/competitions/552/submits?view=submit
    with open(json_path, 'w') as f:
        json.dump(outs, f)  


