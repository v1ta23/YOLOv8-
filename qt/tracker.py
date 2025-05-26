# -*- coding: utf-8 -*-
# file: tracker.py
import numpy as np
from collections import deque
import time # 导入 time 模块用于计时

def iou(bbox1, bbox2):
    """计算两个边界框的 IoU"""
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0

class SimpleTrack:
    """代表一个简单的跟踪目标"""
    def __init__(self, track_id, bbox, frame_id):
        self.id = track_id
        self.bbox = np.array(bbox)
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.history = deque(maxlen=10)
        self.history.append(self.bbox)
        self.predicted_bbox = self.bbox.copy()

    def update(self, bbox, frame_id):
        self.bbox = np.array(bbox)
        self.hits += 1
        self.time_since_update = 0
        self.history.append(self.bbox)

    def predict(self):
        self.predicted_bbox = self.bbox.copy()
        self.age += 1
        self.time_since_update += 1
        return self.predicted_bbox

    def get_state(self):
        return self.bbox

class SimpleSortTracker:
    """简化版 SORT 跟踪器"""
    def __init__(self, max_age=5, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 0
        self.frame_count = 0

    def update(self, detections):
        """
        更新跟踪器状态
        detections: [[x1, y1, x2, y2, conf, cls], ...] 或空列表
        返回: [[x1, y1, x2, y2, track_id], ...] 或空数组
        """
        self.frame_count += 1
        print(f"  [Tracker Update Frame: {self.frame_count}] Start. Num tracks: {len(self.tracks)}, Num detections: {len(detections)}") # 追踪器入口

        start_time = time.time() # 记录开始时间

        # 1. 预测现有跟踪的位置
        predicted_bboxes = np.zeros((len(self.tracks), 4))
        active_track_indices = [] # 需要预测的 track 的索引
        for i, track in enumerate(self.tracks):
            predicted_bboxes[i, :] = track.predict()
            active_track_indices.append(i)
        print(f"  [Tracker] Predicted {len(predicted_bboxes)} tracks.")

        # 2. 关联检测和预测
        matched_indices = [] # (det_idx, track_real_idx)
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks))) # 用 track 在 self.tracks 中的原始索引

        if detections and self.tracks: # 只有同时有检测和跟踪时才进行匹配
            print("  [Tracker] Calculating IoU matrix...")
            iou_matrix = np.zeros((len(detections), len(self.tracks)))
            for d_idx, det in enumerate(detections):
                for t_idx, trk_idx in enumerate(active_track_indices): # 使用 active_track_indices
                    iou_matrix[d_idx, t_idx] = iou(det[:4], predicted_bboxes[t_idx, :])

            print(f"  [Tracker] IoU matrix calculated. Shape: {iou_matrix.shape}. Max IoU: {np.max(iou_matrix) if iou_matrix.size > 0 else 'N/A'}")

            # 简单的贪心匹配
            matched_det_indices = set()
            matched_trk_indices = set()
            match_count = 0

            # 将 IOU 矩阵和索引扁平化以便排序
            if iou_matrix.size > 0:
                indices = np.argsort(-iou_matrix.flatten()) # 降序索引
                rows, cols = np.unravel_index(indices, iou_matrix.shape) # 行(det), 列(track)

                print("  [Tracker] Starting greedy matching...")
                for r, c in zip(rows, cols):
                    # 检查是否已匹配过
                    if r in matched_det_indices or c in matched_trk_indices:
                        continue

                    # 检查 IoU 是否满足阈值
                    if iou_matrix[r, c] >= self.iou_threshold:
                        print(f"    [Match] Found: Det {r} <-> Track {active_track_indices[c]} with IoU {iou_matrix[r, c]:.3f}")
                        matched_indices.append((r, active_track_indices[c])) # 存原始 track 索引
                        matched_det_indices.add(r)
                        matched_trk_indices.add(c)
                        match_count += 1
                    else:
                        # 因为是降序排列，一旦低于阈值，后续的也必然低于阈值
                        # print(f"    [Match] IoU {iou_matrix[r, c]:.3f} below threshold {self.iou_threshold}. Stopping matching.")
                        break # 提前终止循环

            unmatched_detections = list(set(range(len(detections))) - matched_det_indices)
            unmatched_tracks = list(set(active_track_indices) - matched_trk_indices) # 使用 active_track_indices
            print(f"  [Tracker] Matching finished. Matches: {match_count}, Unmatched dets: {len(unmatched_detections)}, Unmatched tracks: {len(unmatched_tracks)}")

        # 3. 更新匹配到的跟踪
        print("  [Tracker] Updating matched tracks...")
        for d_idx, t_idx in matched_indices:
            # t_idx 现在是 self.tracks 中的真实索引
            self.tracks[t_idx].update(detections[d_idx][:4], self.frame_count)

        # 4. 创建新的跟踪
        print("  [Tracker] Creating new tracks for unmatched detections...")
        for d_idx in unmatched_detections:
            # 检查置信度是否足够高 (虽然前面过滤过，但有时会在这里再加一层逻辑)
            # if detections[d_idx][4] >= some_high_threshold: # 例如可以只对高置信度检测创建新轨迹
            new_track = SimpleTrack(self.next_id, detections[d_idx][:4], self.frame_count)
            self.tracks.append(new_track)
            print(f"    [New Track] Created Track ID: {self.next_id}")
            self.next_id += 1

        # 5. 清理旧的跟踪并准备输出
        print("  [Tracker] Cleaning up tracks and preparing output...")
        output_tracks = []
        valid_track_indices = []
        for i, track in enumerate(self.tracks):
            # track 是否应该继续存在
            if track.time_since_update <= self.max_age:
                valid_track_indices.append(i)
                # track 是否应该被输出 (达到最小命中且近期更新)
                # 注意: time_since_update 在 predict() 中增加, 在 update() 中清零
                # 所以刚更新过的 track.time_since_update 是 0
                if track.hits >= self.min_hits and track.time_since_update == 0: # 只输出当前帧更新过的可靠轨迹
                     output_tracks.append(np.append(track.get_state(), track.id))
                elif track.hits < self.min_hits and track.time_since_update == 0:
                     print(f"    [Track {track.id}] Updated but hits ({track.hits}) < min_hits ({self.min_hits}). Not outputting yet.")
                # elif track.time_since_update > 0: # 仅预测未更新的
                #     print(f"    [Track {track.id}] Not updated in this frame (time_since_update={track.time_since_update}).")


        self.tracks = [self.tracks[i] for i in valid_track_indices] # 更新跟踪列表
        print(f"  [Tracker] Outputting {len(output_tracks)} tracks. Kept {len(self.tracks)} tracks total.")

        end_time = time.time()
        print(f"  [Tracker Update Frame: {self.frame_count}] End. Time taken: {end_time - start_time:.4f} seconds.") # 追踪器结束和耗时

        return np.array(output_tracks) if output_tracks else np.empty((0, 5))