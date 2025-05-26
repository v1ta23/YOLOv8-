# -*- coding: utf-8 -*-
# file: utils.py
import os
import cv2
import torch
from PyQt5.QtWidgets import QSplashScreen, QDesktopWidget
from PyQt5.QtGui import QPixmap, QImage, QFont, QPainter, QColor, QFontDatabase
from PyQt5.QtCore import Qt

def load_yolo_model(path):
    """加载 YOLO 模型"""
    if path is None or not os.path.exists(path):
        print(f"模型路径无效或文件不存在: {path}")
        return None
    try:
        # 假设使用 ultralytics 库
        from ultralytics import YOLO
        model = YOLO(path)
        print(f"模型加载成功: {path}")
        if hasattr(model, 'names'):
            print("模型类别:", model.names)
        else:
            print("模型类别名称不可用。")
        return model
    except ImportError:
        print("错误：无法导入 Ultralytics 库。请确保已安装: pip install ultralytics")
        return None
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None

def centerWindow(window):
    """将窗口居中显示"""
    qr = window.frameGeometry()
    cp = QDesktopWidget().availableGeometry().center()
    qr.moveCenter(cp)
    window.move(qr.topLeft())

def displayImage(label_widget, pixmap):
    """按比例缩放并显示 QPixmap 到 QLabel"""
    if pixmap.isNull():
        label_widget.setText("无效图像")
        return
    scaled_pixmap = pixmap.scaled(label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
    label_widget.setPixmap(scaled_pixmap)

def displayCvImage(label_widget, cv_img):
    """将 OpenCV 图像 (BGR) 转换为 QPixmap 并显示"""
    try:
        if cv_img is None or cv_img.size == 0:
            label_widget.setText("空图像帧")
            return
        h, w = cv_img.shape[:2]
        # 检查图像通道数
        if len(cv_img.shape) == 3 and cv_img.shape[2] == 3: # 彩色图像 BGR
            rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            qt_image = QImage(rgb_image.data, w, h, 3 * w, QImage.Format_RGB888)
        elif len(cv_img.shape) == 2: # 灰度图像
            qt_image = QImage(cv_img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            label_widget.setText("不支持的图像格式")
            return

        pixmap = QPixmap.fromImage(qt_image)
        displayImage(label_widget, pixmap) # 复用 displayImage 进行缩放显示
    except Exception as e:
        print(f"显示CV图像时出错: {e}")
        label_widget.setText("显示错误")

class CustomSplashScreen(QSplashScreen):
    """自定义启动画面"""
    def __init__(self, pixmap, flags=Qt.WindowStaysOnTopHint):
        super().__init__(pixmap, flags)
        self.message = ""

    def setMessage(self, message):
        self.message = message
        # 设置消息显示属性：底部居中，白色字体
        self.showMessage(self.message, Qt.AlignBottom | Qt.AlignHCenter, Qt.white)

def get_monospace_font():
    """尝试获取一个可用的等宽字体名称"""
    monospace_font = "Courier New" # 默认
    try:
        db = QFontDatabase()
        available_fonts = db.families()
        # 优先选择的等宽字体列表
        preferred_mono = ['Fixedsys', 'Consolas', 'Monaco', 'DejaVu Sans Mono', 'Courier New']
        for font in preferred_mono:
            if font in available_fonts:
                monospace_font = font
                print(f"UI 将使用系统等宽字体: {monospace_font}")
                break
    except Exception as e:
        print(f"查找等宽字体时出错: {e}, 将使用默认字体 {monospace_font}")
    return monospace_font