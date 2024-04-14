# 引入各项包
# 用于处理文件路径的对象
import sys
from pathlib import Path
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import tempfile
import os
import glob

# 模型路径
# 获取当前文件的绝对路径
file_path = Path(__file__).resolve()
# 获取当前文件的父目录，即根目录的路径
root_path = file_path.parent
# 检查根目录是否已经在sys.path列表中，如果不在，则执行下面的代码
if root_path not in sys.path:
    # 将根目录的路径添加到sys.path列表中
    sys.path.append(str(root_path))
# 获取根目录相对于当前工作目录的绝对路径
root = root_path.relative_to(Path.cwd())
print(root)

# 深度学习模型配置
Detection_model_dir = root / 'weights' / 'detection'
Segmentation_model_dir = root / 'weights' / 'segmentation'
Pose_model_dir = root / 'weights' / 'pose'
#自动读取模型目录下的所有.pt文件
Detection_model_names=[file.name for file in Detection_model_dir.glob('*.pt')]
Segmentation_model_names=[file.name for file in Segmentation_model_dir.glob('*.pt')]
Pose_model_names=[file.name for file in Pose_model_dir.glob('*.pt')]

Detection_model_list = Detection_model_names
Segmentation_model_list = Segmentation_model_names
Pose_model_list = Pose_model_names


# 加载模型
@st.cache_resource
def load_model(model_path):
    model = YOLO(model_path)
    return model

#显示检测到的帧
def _display_detected_frames(conf,model,st_frame,image):
    #将输入图像调整大小，宽度为720，宽高比为16：9
    image = cv2.resize(image,(720,int(720 * (9/16))))
    #调用模型的Predict方法进行检测
    res = model.predict(image,conf=conf)
    #将检测结果中的第一个结果进行绘制
    res_plotted = res[0].plot()
    #显示检测结果，caption是标题，use_colume_width=True代表使用整个列的宽度来显示，使得图像大小适应屏幕宽度
    st_frame.image(res_plotted,caption="视频检测结果",channels="BGR",use_column_width=True)

# 图像处理
def infer_uploaded_image(conf, model):
    #上传图片的小组件，允许用户上传图像，支持JPG,PNG,BMP等格式的图像文件
    source_img = st.sidebar.file_uploader(
        label="请上传",
        type=("jpg", "jpeg", "png", "bmp", "webp")
    )
    #将界面分为两列
    col1, col2 = st.columns(2)
    #在第一列显示原始上传图像
    with col1:
        #如果上传了图像文件，则打开图像文件并显示
        if source_img:
            uploaded_image = Image.open(source_img)
            st.image(image=source_img,
                     caption="原图片",
                     use_column_width=True)
    #如果上传了图像，则
    if source_img:
        #如果用户点击了，则执行对象检测
        if st.button("执行"):
            #加载动画，告知用户程序正在处理中
            with st.spinner("推理中…………"):
                #调用模型的Predict方法，对上传的图像进行预测
                res = model.predict(uploaded_image, conf=conf)
                #获取检测对象的坐标信息
                boxes = res[0].boxes
                #绘制检测结果，并翻译图像通道顺序
                res_plotted = res[0].plot()[:, :, ::-1]
                #第二列显示绘制后的检测图像
                with col2:
                    st.image(res_plotted, caption="检测图像", use_column_width=True)
                    #用于捕获异常并显示
                    try:
                        #展开部件，用于查看检测到的对象的详细结果
                        with st.expander("检测结果"):
                            #遍历所有检测到的对象框，显示坐标信息
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        st.write("没有上传图像")
                        st.write(ex)

#视频处理
def infer_uploaded_video(conf,model):
    source_video = st.sidebar.file_uploader(
        label="请上传"
    )
    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("执行"):
            with st.spinner("推理中"):
                try:
                    tfile = tempfile.NamedTemporaryFile()
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(
                        tfile.name)
                    st_frame = st.empty()
                    while (vid_cap.isOpened()):
                        success,image = vid_cap.read()
                        if success:
                            _display_detected_frames(conf,model,st_frame,image)
                        else:
                            vid_cap.release()
                            break
                except Exception as e:
                    st.error(f"视频错误：{e}")

# 设置页面
st.set_page_config(
    # 设置浏览器页面标题
    page_title="深度学习项目-人车目标检测",
    # 设置页面布局为wide，页面宽度会被扩展以填满整个屏幕的宽度
    layout="wide",
    # 默认情况下侧边栏展开，显示所有内容
    initial_sidebar_state="expanded"
)

# 创建页面标题
st.title("CV项目人车目标检测")

# 侧边栏
st.sidebar.header('模型参数')

# 选择任务
task_type = st.sidebar.selectbox("选择任务", ["Detection", "Segment", "Pose"])

model_type = None
if task_type == "Detection":
    model_type = st.sidebar.selectbox("选择模型", Detection_model_list)
elif task_type == "Segment":
    model_type = st.sidebar.selectbox("选择模型", Segmentation_model_list)
elif task_type == "Pose":
    model_type = st.sidebar.selectbox("选择模型", Pose_model_list)
else:
    st.error("选择的任务类型无效")

# 置信度
confidence = float(st.sidebar.slider("选择模型置信度", 10, 100, 50)) / 100
# 初始化为空字符串
model_path = ""
# 如果用户已经选择了模型类型，则根据任务类型和模型类型构造模型的路径
if model_type:
    if task_type == "Detection":
        model_path = Path(Detection_model_dir, str(model_type))
        print(f"任务类型：{model_type}")
        print(f"模型目录：{Detection_model_dir}")
        print(f"模型路径：{model_path}")
    elif task_type == "Segment":
        model_path = Path(Segmentation_model_dir, str(model_type))
    elif task_type == "Pose":
        model_path = Path(Pose_model_dir, str(model_type))
    else:
        st.error("请选择模型")  # 如果为空，则提示请选择模型

# 加载模型
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"无法加载模型，请检查路径：{model_path}")

# 图片/视频选择
st.sidebar.header("图片/视频")
source_list = ["图片", "视频"]
source_selectbox = st.sidebar.selectbox(
    "请选择", source_list)

# 上传图片视频
source_img = None
if source_selectbox == source_list[0]:
    infer_uploaded_image(confidence, model)
elif source_selectbox == source_list[1]:
    infer_uploaded_video(confidence, model)
else:
    st.error("目前只支持图片和视频")