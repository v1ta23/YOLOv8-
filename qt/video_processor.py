# -*- coding: utf-8 -*-
# file: video_processor.py
import cv2
import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QWaitCondition # 导入 QMutex 和 QWaitCondition (更优的暂停方式，但先用简单方式)

from tracker import SimpleSortTracker

class VideoProcessingThread(QThread):
    frame_processed = pyqtSignal(np.ndarray, int, float)
    processing_finished = pyqtSignal(str)
    update_progress = pyqtSignal(int)

    def __init__(self, video_path, model, confidence_threshold=0.4):
        super().__init__()
        self.video_path = video_path
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.tracker = SimpleSortTracker()
        self._is_running = True
        self._is_paused = False # 添加暂停标志
        self.frame_num = 0
        # --- 更优的暂停方式 (可选，如果简单方式效果不好可以替换) ---
        # self.pause_mutex = QMutex()
        # self.pause_cond = QWaitCondition()
        # ----------------------------------------------------------

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.processing_finished.emit("错误：无法打开视频文件")
            return

        try: frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except: frame_count = -1
        processed_count = 0

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_num = 0

        while self._is_running:
            # --- 检查暂停状态 (简单方式) ---
            while self._is_paused:
                self.msleep(100) # 暂停时休眠 100 毫秒
                # 如果在暂停时也需要能响应停止信号，需要在这里加判断
                if not self._is_running:
                    break # 如果在暂停期间被要求停止，则退出内层循环
            if not self._is_running: # 再次检查停止信号
                break # 退出外层循环
            # --- ----------------------- ---

            # --- 更优的暂停方式 (可选) ---
            # self.pause_mutex.lock()
            # while self._is_paused:
            #     self.pause_cond.wait(self.pause_mutex) # 等待 resume() 信号
            # self.pause_mutex.unlock()
            # if not self._is_running: break # 检查停止
            # --- ----------------------- ---


            ret, frame = cap.read()
            if not ret: break
            self.frame_num += 1

            # print(f"\n--- 处理帧: {self.frame_num} ---")

            start_time = time.time()
            detections_raw = self.run_inference(frame)

            detections_filtered = []
            max_confidence = 0.0
            if detections_raw is not None and len(detections_raw) > 0:
                try:
                    for *xyxy, conf, cls in detections_raw:
                        current_conf = float(conf)
                        if current_conf >= self.confidence_threshold:
                            detections_filtered.append(list(xyxy) + [current_conf, cls])
                            if current_conf > max_confidence:
                                max_confidence = current_conf
                except (TypeError, ValueError) as e: print(f"[{self.frame_num}] 处理检测结果时出错: {e}")

            try:
                tracked_objects = self.tracker.update(detections_filtered)
            except Exception as e: print(f"[{self.frame_num}] Tracker Update Error: {e}"); tracked_objects = np.empty((0, 5))

            processed_frame = frame.copy()
            tracking_count = len(tracked_objects) if tracked_objects is not None else 0

            if tracked_objects is not None and tracking_count > 0 :
                for *xyxy, track_id in tracked_objects:
                    # ... (绘制代码不变) ...
                    x1, y1, x2, y2 = map(int, xyxy); label = f'ID: {int(track_id)}'; color = (255, 0, 0)
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    (lw, lh), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2); l_ymin = max(y1 - lh - 5, 0)
                    cv2.rectangle(processed_frame, (x1, l_ymin), (x1 + lw, y1), color, cv2.FILLED); cv2.putText(processed_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)


            end_time = time.time()
            process_time_ms = (end_time - start_time) * 1000
            current_fps = 1000 / process_time_ms if process_time_ms > 0 else float('inf')
            status_text = f"FPS: {current_fps:.1f} | Tracked: {tracking_count}"
            # 如果暂停，在画面上加提示 (可选)
            if self._is_paused:
                 cv2.putText(processed_frame, "PAUSED", (frame_width // 2 - 50, frame_height // 2),
                             cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 255), 2)
            else:
                 cv2.putText(processed_frame, status_text, (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)


            try:
                self.frame_processed.emit(processed_frame, tracking_count, max_confidence)
            except Exception as e: print(f"[{self.frame_num}] Error emitting frame_processed signal: {e}")

            processed_count += 1
            if frame_count > 0:
                 progress = int((processed_count / frame_count) * 100)
                 self.update_progress.emit(progress)

            # self.msleep(1) # 暂停逻辑自带休眠，这里可以不需要，或者减少时间

        cap.release()
        status_msg = f"视频处理完成。共处理 {processed_count} 帧。" if self._is_running else "视频处理已由用户中断。"
        self.processing_finished.emit(status_msg)

    def pause(self):
        """暂停处理"""
        print("请求暂停线程...")
        self._is_paused = True
        # --- 更优的暂停方式 ---
        # self.pause_mutex.lock()
        # self._is_paused = True
        # self.pause_mutex.unlock()
        # --- --------------- ---

    def resume(self):
        """恢复处理"""
        print("请求恢复线程...")
        self._is_paused = False
        # --- 更优的暂停方式 ---
        # self.pause_mutex.lock()
        # self._is_paused = False
        # self.pause_mutex.unlock()
        # self.pause_cond.wakeAll() # 唤醒等待的线程
        # --- --------------- ---

    def stop(self):
        """请求停止线程 (同时需要解除暂停状态以允许循环退出)"""
        print("视频处理线程停止请求。")
        self._is_running = False
        if self._is_paused: # 如果当前是暂停状态，需要先恢复才能让循环判断 _is_running
             self.resume()
        # --- 更优的暂停方式也需要唤醒 ---
        # self.pause_cond.wakeAll()
        # --- ------------------------ ---

    # run_inference 方法保持不变...
    def run_inference(self, frame):
        if self.model is None: return None
        try:
            results = self.model(frame, verbose=False); detections = []
            if results and isinstance(results, list) and len(results) > 0:
                 if hasattr(results[0], 'boxes') and results[0].boxes is not None: detections = results[0].boxes.data.cpu().numpy()
            elif hasattr(results, 'pred') and results.pred is not None:
                 pred = results.pred[0];
                 if pred is not None: detections = pred.cpu().numpy()
            return detections
        except Exception as e: print(f"模型推理时出错: {e}"); return None