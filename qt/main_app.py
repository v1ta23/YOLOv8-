# -*- coding: utf-8 -*-
# file: main_app.py
import sys
import os
import time
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtGui import QPixmap, QPainter, QFont, QColor
from PyQt5.QtCore import Qt
# 从其他模块导入必要的类和函数
from ui_main_window import DroneDetectionApp
from utils import CustomSplashScreen

def run_application():
    """运行应用程序的主函数"""
    # --- 设置高 DPI 支持 ---
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # --- 创建和显示启动画面 ---
    try:
        # 尝试创建启动画面（如果需要）
        splash_pix = QPixmap(600, 300)
        splash_pix.fill(QColor(40, 44, 52)) # 深色背景
        painter = QPainter(splash_pix)
        painter.setPen(QColor(200, 200, 200))
        font = QFont("Segoe UI", 24, QFont.Bold)
        painter.setFont(font)
        painter.drawText(splash_pix.rect(), Qt.AlignCenter, "无人机检测系统\n正在启动...")
        painter.end()
        splash = CustomSplashScreen(splash_pix)
        splash.show()
        splash.setMessage("初始化...")
        app.processEvents() # 确保启动画面显示
    except Exception as e:
        print(f"创建启动画面时出错: {e}")
        splash = None # 出错则不使用启动画面

    # --- 确定模型文件路径 ---
    model_load_status = False
    model_file_path = None
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 注意这里的相对路径是相对于 main_app.py 的位置
        # 假设 runs_ultralytics 文件夹在 main_app.py 的同级或上级目录
        # 您可能需要根据实际文件结构调整这里的路径
        model_relative_path = os.path.join('runs_ultralytics', 'uav_yolov8s_run', 'weights', 'best.pt')
        # 尝试多种可能的路径组合
        possible_paths = [
            os.path.join(script_dir, model_relative_path),
            os.path.join(script_dir, '..', model_relative_path), # 尝试上一级目录
        ]
        for path in possible_paths:
            if os.path.exists(path):
                model_file_path = path
                model_load_status = True
                print(f"找到模型文件: {model_file_path}")
                break

        if splash:
            if model_load_status:
                splash.setMessage("正在加载模型...")
            else:
                splash.setMessage(f"警告：未找到模型文件!")
            app.processEvents()

        if not model_load_status:
            print(f"警告: 未能在预设路径找到模型文件 ({model_relative_path})。")
            # 可选：在这里弹窗提示用户
            # QMessageBox.warning(None, "模型未找到", f"未能在以下路径找到模型文件:\n{possible_paths}\n检测功能将不可用。")
            if splash: time.sleep(2) # 如果有启动画面，暂停一会显示警告

    except Exception as e:
        print(f"确定模型路径时出错: {e}")
        if splash: splash.setMessage("模型加载出错!")
        app.processEvents()
        if splash: time.sleep(2)

    # --- 创建主窗口 ---
    if splash:
        splash.setMessage("正在初始化界面...")
        app.processEvents()

    try:
        mainWin = DroneDetectionApp(model_path=model_file_path, model_loaded_ok=model_load_status)
        mainWin.show()
        if splash:
            splash.finish(mainWin) # 主窗口显示后关闭启动画面
    except Exception as e:
        print(f"创建主窗口时出错: {e}")
        QMessageBox.critical(None, "程序错误", f"无法初始化主窗口: {e}")
        sys.exit(1) # 严重错误，退出

    # --- 启动事件循环 ---
    sys.exit(app.exec_())

if __name__ == '__main__':
    run_application()