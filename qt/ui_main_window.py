# -*- coding: utf-8 -*-
# file: ui_main_window.py
# ... (其他 import 和类定义保持不变) ...
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QPushButton, QLabel, QVBoxLayout,
                             QHBoxLayout, QWidget, QFileDialog, QMessageBox,
                             QSizePolicy, QGridLayout)
from PyQt5.QtGui import QPixmap, QFontDatabase
from PyQt5.QtCore import Qt, QTimer
from video_processor import VideoProcessingThread
from utils import load_yolo_model, centerWindow, displayImage, displayCvImage, get_monospace_font

class DroneDetectionApp(QMainWindow):
    # ... (__init__, initUI 前半部分不变) ...
    def __init__(self, model_path=None, model_loaded_ok=False):
        super().__init__()
        self.setWindowTitle("无人机检测系统")
        self.setGeometry(50, 50, 1400, 850)
        centerWindow(self)
        self.model = None; self.model_loaded_ok = model_loaded_ok; self.confidence_threshold = 0.4
        if model_path and self.model_loaded_ok:
             self.model = load_yolo_model(model_path)
             if self.model is None: self.model_loaded_ok = False; print("警告: 模型加载失败...")
        self.current_file_path = None; self.is_video = False; self.video_thread = None; self.is_paused = False
        self.initUI()
        self.applyStyles()

    def initUI(self):
        """初始化用户界面"""
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        grid_layout = QGridLayout(central_widget)
        grid_layout.setSpacing(15)

        # --- 顶部控制按钮 ---
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget); button_layout.setContentsMargins(5, 5, 5, 5); button_layout.setSpacing(10)
        self.btn_load_image = QPushButton("加载图片"); self.btn_load_image.setObjectName("neumorphicButton"); self.btn_load_image.clicked.connect(self.loadImage)
        self.btn_load_video = QPushButton("加载视频"); self.btn_load_video.setObjectName("neumorphicButton"); self.btn_load_video.clicked.connect(self.loadVideo)
        self.btn_start_detection = QPushButton("开始检测"); self.btn_start_detection.setObjectName("neumorphicButton"); self.btn_start_detection.clicked.connect(self.startDetection); self.btn_start_detection.setEnabled(False)
        self.btn_pause_resume = QPushButton("暂停"); self.btn_pause_resume.setObjectName("neumorphicButton"); self.btn_pause_resume.clicked.connect(self.togglePauseResume); self.btn_pause_resume.setEnabled(False)
        self.btn_stop_detection = QPushButton("停止检测"); self.btn_stop_detection.setObjectName("neumorphicButton"); self.btn_stop_detection.clicked.connect(self.stopPotentialVideoThread); self.btn_stop_detection.setEnabled(False)
        button_layout.addWidget(self.btn_load_image); button_layout.addWidget(self.btn_load_video); button_layout.addStretch(1); button_layout.addWidget(self.btn_start_detection); button_layout.addWidget(self.btn_pause_resume); button_layout.addWidget(self.btn_stop_detection)

        # --- 中间显示区域 ---
        display_widget = QWidget(); display_widget.setObjectName("displayArea")
        display_layout = QHBoxLayout(display_widget); display_layout.setSpacing(10)
        original_widget = QWidget(); original_widget.setObjectName("transparentWidget"); original_layout = QVBoxLayout(original_widget)
        lbl_original_title = QLabel("原始画面"); lbl_original_title.setAlignment(Qt.AlignCenter)
        self.lbl_original_display = QLabel("请加载图片或视频"); self.lbl_original_display.setAlignment(Qt.AlignCenter); self.lbl_original_display.setMinimumSize(600, 450); self.lbl_original_display.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored); self.lbl_original_display.setObjectName("displayLabel")
        original_layout.addWidget(lbl_original_title); original_layout.addWidget(self.lbl_original_display, 1)
        result_widget = QWidget(); result_widget.setObjectName("transparentWidget"); result_layout = QVBoxLayout(result_widget)
        lbl_result_title = QLabel("检测/跟踪结果"); lbl_result_title.setAlignment(Qt.AlignCenter)
        self.lbl_result_display = QLabel("结果将显示在这里"); self.lbl_result_display.setAlignment(Qt.AlignCenter); self.lbl_result_display.setMinimumSize(600, 450); self.lbl_result_display.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored); self.lbl_result_display.setObjectName("displayLabel")
        result_layout.addWidget(lbl_result_title); result_layout.addWidget(self.lbl_result_display, 1)
        display_layout.addWidget(original_widget, 1); display_layout.addWidget(result_widget, 1)

        # --- 底部信息显示屏 ---
        self.info_widget = QWidget() # 将 info_widget 设为 self 属性，方便调试
        info_layout = QVBoxLayout(self.info_widget)
        self.info_widget.setObjectName("infoPanelStaticNoise") # 使用这个 objectName
        self.info_widget.setFixedHeight(120)
        info_layout.setContentsMargins(20, 10, 20, 10); info_layout.setSpacing(5)
        info_line1_layout = QHBoxLayout()
        self.lbl_file_info_title = QLabel("文件:"); self.lbl_file_info_title.setObjectName("infoTitleNoise")
        self.lbl_file_info = QLabel("N/A"); self.lbl_file_info.setObjectName("infoDataNoise"); self.lbl_file_info.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        info_line1_layout.addWidget(self.lbl_file_info_title); info_line1_layout.addWidget(self.lbl_file_info, 1)
        info_line2_layout = QHBoxLayout()
        self.lbl_status_title = QLabel("状态:"); self.lbl_status_title.setObjectName("infoTitleNoise")
        self.lbl_status = QLabel("就绪"); self.lbl_status.setObjectName("infoDataNoise")
        self.lbl_detection_count_title = QLabel("数量:"); self.lbl_detection_count_title.setObjectName("infoTitleNoise"); self.lbl_detection_count_title.setAlignment(Qt.AlignRight)
        self.lbl_detection_count = QLabel("N/A"); self.lbl_detection_count.setObjectName("infoDataNoise"); self.lbl_detection_count.setMinimumWidth(60); self.lbl_detection_count.setAlignment(Qt.AlignRight)
        self.lbl_max_conf_title = QLabel("置信度:"); self.lbl_max_conf_title.setObjectName("infoTitleNoise"); self.lbl_max_conf_title.setAlignment(Qt.AlignRight)
        self.lbl_max_confidence = QLabel("N/A"); self.lbl_max_confidence.setObjectName("infoDataNoise"); self.lbl_max_confidence.setMinimumWidth(80); self.lbl_max_confidence.setAlignment(Qt.AlignRight)
        info_line2_layout.addWidget(self.lbl_status_title); info_line2_layout.addWidget(self.lbl_status, 1); info_line2_layout.addSpacing(20)
        info_line2_layout.addWidget(self.lbl_detection_count_title); info_line2_layout.addWidget(self.lbl_detection_count); info_line2_layout.addSpacing(20)
        info_line2_layout.addWidget(self.lbl_max_conf_title); info_line2_layout.addWidget(self.lbl_max_confidence)
        info_layout.addLayout(info_line1_layout); info_layout.addLayout(info_line2_layout); info_layout.addStretch()

        # --- 添加到网格布局 ---
        grid_layout.addWidget(button_widget, 0, 0, 1, 1)
        grid_layout.addWidget(display_widget, 1, 0, 1, 1)
        grid_layout.addWidget(self.info_widget, 2, 0, 1, 1) # 使用 self.info_widget
        grid_layout.setRowStretch(0, 0); grid_layout.setRowStretch(1, 1); grid_layout.setRowStretch(2, 0)

        self.update_button_states()

    def applyStyles(self):
        """ QSS 样式 """
        monospace_font = get_monospace_font()

        # 主要颜色定义
        base_color = "rgba(255, 255, 255, 255)"  # 纯白色底色
        shadow_light = "rgba(255, 255, 255, 180)"
        shadow_dark = "rgba(190, 200, 210, 180)"
        button_text_color = "#555577"
        text_color = "#444455"
        title_color = "#333344"

        # 显示屏字体颜色 -
        display_text_color = "#2c3e50"  #
        display_title_color = "#34495e"  #
        display_data_color = "#16a085"  #

        self.setStyleSheet(f"""
            QMainWindow {{ background-color: qlineargradient(spread:pad, x1:0.5, y1:0, x2:0.5, y2:1, stop:0 {base_color}, stop:1 rgba(245, 240, 250, 245)); }}
            QWidget {{ color: {text_color}; font-size: 10pt; font-family: 'Segoe UI', 'Microsoft YaHei', Arial, sans-serif; }}

            /* 按钮样式  */
            QPushButton#neumorphicButton {{ background-color: {base_color}; color: {button_text_color}; border: none; border-radius: 8px; padding: 10px 18px; min-width: 90px; font-weight: bold; border-top: 1px solid {shadow_light}; border-left: 1px solid {shadow_light}; border-bottom: 1px solid {shadow_dark}; border-right: 1px solid {shadow_dark}; }}
            QPushButton#neumorphicButton:hover {{ background-color: rgba(240, 245, 250, 250); }}
            QPushButton#neumorphicButton:pressed {{ background-color: rgba(225, 230, 235, 250); border-top: 1px solid {shadow_dark}; border-left: 1px solid {shadow_dark}; border-bottom: 1px solid {shadow_light}; border-right: 1px solid {shadow_light}; }}
            QPushButton#neumorphicButton:disabled {{ background-color: rgba(225, 230, 235, 180); color: #aaaaaa; border-top: 1px solid rgba(245, 245, 245, 100); border-left: 1px solid rgba(245, 245, 245, 100); border-bottom: 1px solid rgba(200, 210, 220, 100); border-right: 1px solid rgba(200, 210, 220, 100); }}

            QWidget#displayArea {{ background: transparent; border: none; }}
            QWidget#transparentWidget {{ background: transparent; }}

            /* 视频显示框凹陷效果  */
            QLabel#displayLabel {{ background-color: rgba(210, 215, 220, 180); border-width: 2px; border-style: inset; border-color: {shadow_dark}; border-radius: 6px; }}
            QVBoxLayout > QLabel[alignment="AlignCenter"] {{ font-weight: bold; color: {title_color}; font-size: 12pt; padding-bottom: 8px; }}

            /* --- 修改后的显示屏样式：黑色厚圆弧框，纯白底色 --- */
            QWidget#infoPanelStaticNoise {{
                background-color: #ffffff; /* 纯白色背景 */
                border: 8px solid #000000; /* 黑色厚边框 */
                border-radius: 15px; /* 大圆角 */
                margin: 5px;
                padding: 10px; /* 增加内边距，避免文字太贴边 */
            }}

            /* --- 清新可爱的字体样式 --- */
            QWidget#infoPanelStaticNoise QLabel {{
                background-color: transparent; /* 内部标签背景透明 */
                color: {display_text_color}; 
                font-family: "Comic Sans MS", "Segoe UI", "{monospace_font}", sans-serif; /* 更可爱的字体 */
                font-size: 11pt; /* 基础字体大小 */
                padding: 2px 4px;
                /* 字体阴影使文字更清晰 */
                text-shadow: 0.5px 0.5px 1px rgba(150, 150, 150, 0.5);
            }}

            QWidget#infoPanelStaticNoise QLabel#infoTitleNoise {{
                color: {display_title_color}; /* 蓝色标题 */
                font-weight: bold;
                font-size: 12pt; /* 标题字体略大 */
                margin-bottom: 2px;
            }}

            QWidget#infoPanelStaticNoise QLabel#infoDataNoise {{
                color: {display_data_color}; /* 粉色数据显示 */
                font-size: 14pt; /* 数据字体更大 */
                font-weight: bold;
                letter-spacing: 0.5px; /* 字母间距增加可读性 */
            }}
            /* --- 结束 信息显示屏样式 --- */
        """)

    # --- 添加 showEvent 用于调试打印控件尺寸 ---
    def showEvent(self, event):
        """窗口显示事件，在这里打印 info_widget 尺寸"""
        super().showEvent(event)
        # 使用 QTimer.singleShot 确保在布局计算完成后获取尺寸
        QTimer.singleShot(100, lambda: print(f"调试信息: info_widget 尺寸 = {self.info_widget.size()}"))

    # --- 其他方法保持不变 ---
    # ... (update_button_states, togglePauseResume, loadImage, loadVideo, startDetection, processSingleImage, updateVideoResult, onVideoProcessingFinished, stopPotentialVideoThread, closeEvent) ...
    def update_button_states(self, is_processing=False, is_paused=None):
        if is_paused is None: is_paused = self.is_paused
        if is_processing:
            self.btn_load_image.setEnabled(False); self.btn_load_video.setEnabled(False)
            self.btn_start_detection.setEnabled(False); self.btn_stop_detection.setEnabled(True)
            self.btn_pause_resume.setEnabled(self.is_video); self.btn_pause_resume.setText("继续" if is_paused else "暂停")
        else:
            can_start = self.model_loaded_ok and self.current_file_path is not None
            self.btn_load_image.setEnabled(True); self.btn_load_video.setEnabled(True)
            self.btn_start_detection.setEnabled(can_start); self.btn_stop_detection.setEnabled(False)
            self.btn_pause_resume.setEnabled(False); self.btn_pause_resume.setText("暂停")
    def togglePauseResume(self):
        if self.video_thread and self.video_thread.isRunning():
            if self.is_paused:
                self.video_thread.resume(); self.is_paused = False; self.lbl_status.setText("正在跟踪...")
            else:
                self.video_thread.pause(); self.is_paused = True; self.lbl_status.setText("已暂停")
            self.update_button_states(is_processing=True, is_paused=self.is_paused)
    def loadImage(self):
        self.stopPotentialVideoThread(); self.is_paused = False
        options = QFileDialog.Options(); fileName, _ = QFileDialog.getOpenFileName(self, "选择图片文件", "", "图片文件 (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if fileName:
            self.current_file_path = fileName; self.is_video = False
            self.lbl_file_info.setText(os.path.basename(fileName))
            self.lbl_status.setText("图片已加载"); self.lbl_detection_count.setText("N/A (单图)"); self.lbl_max_confidence.setText("N/A")
            pixmap = QPixmap(fileName); displayImage(self.lbl_original_display, pixmap)
            self.lbl_result_display.setText("等待检测..."); self.update_button_states()
    def loadVideo(self):
        self.stopPotentialVideoThread(); self.is_paused = False
        options = QFileDialog.Options(); fileName, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)", options=options)
        if fileName:
            self.current_file_path = fileName; self.is_video = True
            self.lbl_file_info.setText(os.path.basename(fileName))
            self.lbl_status.setText("视频已加载"); self.lbl_detection_count.setText("N/A"); self.lbl_max_confidence.setText("N/A")
            cap = cv2.VideoCapture(fileName)
            if cap.isOpened(): ret, frame = cap.read(); cap.release()
            if ret: displayCvImage(self.lbl_original_display, frame)
            else: self.lbl_original_display.setText("无法读取视频帧")
            self.lbl_result_display.setText("等待跟踪..."); self.update_button_states()
    def startDetection(self):
        if not self.current_file_path: QMessageBox.warning(self, "提示", "请先加载图片或视频文件！"); return
        if not self.model_loaded_ok or self.model is None: QMessageBox.critical(self, "错误", "模型未成功加载，无法进行检测。"); return
        self.stopPotentialVideoThread(); self.is_paused = False
        self.lbl_status.setText("处理中..."); self.lbl_detection_count.setText("..."); self.lbl_max_confidence.setText("...")
        self.update_button_states(is_processing=True, is_paused=self.is_paused)
        if self.is_video:
            self.lbl_status.setText("正在跟踪...")
            self.video_thread = VideoProcessingThread(self.current_file_path, self.model, self.confidence_threshold)
            self.video_thread.frame_processed.connect(self.updateVideoResult)
            self.video_thread.update_progress.connect(lambda p: self.lbl_status.setText(f"处理进度: {p}%") if not self.is_paused else None)
            self.video_thread.processing_finished.connect(self.onVideoProcessingFinished)
            self.video_thread.start()
        else:
            self.lbl_status.setText("正在检测...")
            self.processSingleImage()
    def processSingleImage(self):
        max_confidence = 0.0; target_count = 0
        try:
            img = cv2.imread(self.current_file_path);
            if img is None: raise ValueError("无法读取图片文件")
            results = self.model(img, verbose=False); detections = []
            if results and isinstance(results, list) and len(results) > 0:
               if hasattr(results[0], 'boxes') and results[0].boxes is not None: detections = results[0].boxes.data.cpu().numpy()
            processed_img = img.copy()
            if detections is not None:
                valid_detections = [d for d in detections if float(d[4]) >= self.confidence_threshold]; target_count = len(valid_detections)
                if target_count > 0:
                    confidences = [float(d[4]) for d in valid_detections]; max_confidence = np.max(confidences) if confidences else 0.0
                if hasattr(self.model, 'names'):
                    for *xyxy, conf, cls in valid_detections:
                         class_id = int(cls)
                         try:
                             class_name = self.model.names[class_id]; label = f'{class_name} {conf:.2f}'; color = (0, 255, 0)
                             cv2.rectangle(processed_img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), color, 2)
                             (lw, lh), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2); l_ymin = max(int(xyxy[1]) - lh - 5, 0)
                             cv2.rectangle(processed_img, (int(xyxy[0]), l_ymin), (int(xyxy[0]) + lw, int(xyxy[1])), color, cv2.FILLED); cv2.putText(processed_img, label, (int(xyxy[0]), int(xyxy[1]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                         except IndexError: pass
                         except Exception as e: print(f"绘制检测框时出错: {e}")
            displayCvImage(self.lbl_result_display, processed_img)
            self.lbl_status.setText("图片检测完成"); self.lbl_detection_count.setText(f"{target_count}"); self.lbl_max_confidence.setText(f"{max_confidence:.2%}" if max_confidence > 0 else "N/A")
        except Exception as e:
            self.lbl_status.setText(f"错误: {e}"); self.lbl_detection_count.setText("错误"); self.lbl_max_confidence.setText("错误")
            QMessageBox.critical(self, "检测错误", f"处理图片时发生错误: {e}")
        finally: self.update_button_states(is_processing=False)
    def updateVideoResult(self, frame, count, max_conf):
        displayCvImage(self.lbl_result_display, frame)
        self.lbl_detection_count.setText(f"{count}")
        self.lbl_max_confidence.setText(f"{max_conf:.2%}" if max_conf > 0 else "N/A")
    def onVideoProcessingFinished(self, message):
        self.lbl_status.setText(message); self.video_thread = None; self.is_paused = False
        self.update_button_states(is_processing=False)
    def stopPotentialVideoThread(self):
        if self.video_thread and self.video_thread.isRunning():
            print("正在停止视频线程..."); self.video_thread.stop()
            if not self.video_thread.wait(3000): print("警告: 视频线程未能在3秒内正常停止。")
            self.video_thread = None; print("视频处理线程已停止或等待超时。")
            if self.lbl_status.text().startswith("处理中") or self.lbl_status.text().startswith("正在跟踪") or self.lbl_status.text() == "已暂停": self.lbl_status.setText("处理已停止")
        self.is_paused = False; self.update_button_states(is_processing=False)
    def closeEvent(self, event):
        print("正在关闭应用程序..."); self.stopPotentialVideoThread(); event.accept()