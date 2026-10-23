路面垃圾细分类检测数据集，![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/45edaf6c5663407598ff51ca8a875dc7.png)
3321张，yolo，voc，coco三种标注方式
图像尺寸:640*640
类别数量:17类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1dad3631ca8b4a4c9d7ebf01aaf7d307.png)

训练集图像数量:2661; 验证集图像数量:329； 测试集图像数量:331
类别名称: 每一类图像数 ，每一类标注数![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0267f4f63da14151ac70a1a25c22f36b.png)

Plastic Bottle 塑料瓶 457 1260
Cans 易拉罐 250 686
Paper Cup 纸杯 221 453
Cardboard 硬纸板 239 1033
Plastic Bag 塑料袋 253 714
Plastic Wrapper 塑料包装膜 292 2449
Styrofoam 泡沫塑料 130 436
Glass Bottle 玻璃瓶 281 533
Tetra Pack 利乐包装 645 942
Face Mask 口罩 336 977
Paper Bag 纸袋 504 1579
Plastic Cup 塑料杯 307 536
Peel 果皮 39 164
Rags 碎布 80 145
Paperboard 纸板 127 426
Pile of Leaves 落叶堆 148 392
PaperBag 纸袋 5 5
image num: 3321![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2fda7c4ae0ca4ef39eb068a507d5b778.png)


模型代码
采用 YOLOv11n 网络训练
训练轮次：80 个 epoch
提供全部训练 + 测试源代码
训练精度 mAP 效果如图所示![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1c772ef27c7349f488ad0e3d9c4560c5.png)

PyQt5 界面功能![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/82d1bf27a76f400bae8f693ecd6ea6cc.png)

界面使用 PyQt5 开发，提供全部源码（.ui、.qrc、.py 及图标文件）
支持图片检测、视频检测、摄像头实时检测
界面实时显示：目标位置、目标总数、置信度等信息
支持检测结果保存导出
操作简单直观，无需命令行
4、
运行环境：Python=3.8、opencv-python、PyQt5、torch
支持 Windows、Linux 系统![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b489a39f49da4048aa58e5836d6b260a.png)
# 基于YOLOv11路面垃圾细分类检测系统简易代码
## 1、数据集yaml配置 `road_trash.yaml`
```yaml
path: ./road_trash_dataset
train: images/train
val: images/val
test: images/test

names:
  0: Plastic Bottle
  1: Cans
  2: Paper Cup
  3: Cardboard
  4: Plastic Bag
  5: Plastic Wrapper
  6: Styrofoam
  7: Glass Bottle
  8: Tetra Pack
  9: Face Mask
  10: Paper Bag
  11: Peel
  12: Rags
  13: Paperboard
  14: Pile of Leaves
```

## 2、训练脚本 `train_trash.py`
```python
from ultralytics import YOLO

if __name__ == '__main__':
    # 加载YOLOv11n检测模型
    model = YOLO("yolo11n.pt")
    results = model.train(
        data="road_trash.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        project="road_trash_output",
        name="yolo11_road_trash"
    )
```

## 3、图像推理脚本 `predict_trash.py`
```python
from ultralytics import YOLO

model = YOLO("./road_trash_output/yolo11_road_trash/weights/best.pt")
# 单图推理
res = model.predict("test.jpg", save=True, conf=0.5)
for r in res:
    print("检测目标：", r.boxes.cls)
    print("置信度：", r.boxes.conf)
    print("坐标：", r.boxes.xyxy)
```

## 4、极简PyQt界面代码（对应你截图里的路面垃圾检测系统）
```python
import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel,
                             QFileDialog, QTableWidget, QTableWidgetItem)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO

class TrashDetectWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于YOLOv11的路面垃圾细分类检测系统")
        self.setGeometry(100, 100, 1200, 800)
        self.model = YOLO("./road_trash_output/yolo11_road_trash/weights/best.pt")
        self.init_ui()

    def init_ui(self):
        # 图片显示区域
        self.img_label = QLabel("图片预览区", self)
        self.img_label.setGeometry(20, 20, 700, 600)
        self.img_label.setStyleSheet("border:1px solid #444;")
        self.img_label.setAlignment(Qt.AlignCenter)

        # 按钮
        self.btn_open = QPushButton("打开图片", self)
        self.btn_open.setGeometry(750, 40, 150, 40)
        self.btn_open.clicked.connect(self.load_image)

        self.btn_detect = QPushButton("开始检测", self)
        self.btn_detect.setGeometry(750, 100, 150, 40)
        self.btn_detect.clicked.connect(self.run_detect)

        # 结果表格
        self.table = QTableWidget(self)
        self.table.setGeometry(20, 640, 1140, 120)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["序号", "类别", "置信度", "坐标"])

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片(*.jpg *.png *.bmp)")
        if file_path:
            self.img_path = file_path
            self.img_label.setPixmap(QPixmap(file_path).scaled(self.img_label.size(), Qt.KeepAspectRatio))

    def run_detect(self):
        results = self.model(self.img_path)[0]
        img_out = results.plot()
        h, w, c = img_out.shape
        qimg = QImage(img_out.data, w, h, c*w, QImage.Format_RGB888).rgbSwapped()
        self.img_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.img_label.size(), Qt.KeepAspectRatio))

        # 填充表格
        self.table.setRowCount(0)
        for idx, box in enumerate(results.boxes):
            cls_name = self.model.names[int(box.cls)]
            conf = float(box.conf)
            xyxy = [round(x.item(),1) for x in box.xyxy[0]]
            self.table.insertRow(idx)
            self.table.setItem(idx,0, QTableWidgetItem(str(idx+1)))
            self.table.setItem(idx,1, QTableWidgetItem(cls_name))
            self.table.setItem(idx,2, QTableWidgetItem(f"{conf:.3f}"))
            self.table.setItem(idx,3, QTableWidgetItem(f"{xyxy}"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = TrashDetectWindow()
    win.show()
    sys.exit(app.exec_())
```

## 环境安装命令
```bash
pip install ultralytics torch opencv-python pyqt5
```

关键词：路面垃圾检测、多类别垃圾分类、YOLOv11、航拍路面垃圾识别、垃圾细分类目标检测。
