# CarDetect

人车目标检测项目

### 使用方法

- env_init.ipynb：初始化环境
- voc2yolo.py：比赛图片转成YOLO格式
- datasets文件夹 : 数据集
  - cars：测试集
  - cars_v2: 包含手动收集图片的训练集和验证集
  - cars_v3：原始数据集的训练集和验证集 
- cars_train.py : 使用YOLO训练，可调整参数
- cars.yaml : 数据集路径和类别，YOLO调用
- cars_predict.py 推理并转成json格式文件
- app：部署文件

### 贡献原则

- 最新以及最终版本的code保存在main branch  
- 开发时，创建自己的branch，在自己的branch上开发  
- 本地测试通过后，创建pull request，请求merge到main branch  
- pull request需要至少一名其他开发成员review，通过后merge到main  
- 不要直接在main branch上开发，以免引发问题break code!!!  
