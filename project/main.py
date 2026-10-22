# -*- coding: utf-8 -*-
# 解决OMP多dll冲突，必须放在所有import最顶部
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import time
import json
import re
import threading
import webbrowser
import sqlite3
import hashlib
import urllib.parse
from datetime import datetime
from flask import (Flask, Response, render_template, jsonify, request,
                   send_from_directory, redirect, url_for, session)
from ultralytics import YOLO
import numpy as np
import warnings
import logging
import platform

# ===== 音频告警模块（持续蜂鸣）=====
alarm_stop_event = threading.Event()
alarm_thread = None

if platform.system() == "Windows":
    import winsound
    def beep_loop():
        while not alarm_stop_event.is_set():
            try:
                winsound.Beep(1200, 300)
                time.sleep(0.1)
            except Exception:
                break
else:
    import subprocess
    def beep_loop():
        while not alarm_stop_event.is_set():
            try:
                os.system('say "警报" 2>/dev/null &')
            except Exception:
                pass
            time.sleep(1)

def start_alarm():
    global alarm_thread
    if alarm_thread is not None and alarm_thread.is_alive():
        return
    alarm_stop_event.clear()
    alarm_thread = threading.Thread(target=beep_loop, daemon=True)
    alarm_thread.start()

def stop_alarm():
    alarm_stop_event.set()

# 关闭多余日志
warnings.filterwarnings('ignore')
logging.getLogger('ultralytics').setLevel(logging.ERROR)
os.environ['YOLO_VERBOSE'] = 'False'
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# ======================【配置加载】======================
def load_config():
    try:
        with open("config.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

CONFIG = load_config()

CONF_THRESH = float(CONFIG.get("conf_thresh", 0.25))
MASK_THRESH = 0.3
AREA_THRESH = int(CONFIG.get("area_thresh", 80))
ALARM_DURATION_THRESH = float(CONFIG.get("alarm_duration_thresh", 6))
# 危险会话合并窗口（秒）：同一人在该窗口内多次进入警戒区，合并为一次告警记录并累计次数/时长
REENTRY_WINDOW = float(CONFIG.get("reentry_window", 300))
# 白名单人脸匹配阈值（LBP 卡方距离，越小越严格；识别到白名单人员时不报警）
FACE_MATCH_THRESH = float(CONFIG.get("face_match_threshold", 0.42))
FIXED_W, FIXED_H = 640, 480
FRAME_SKIP = int(CONFIG.get("frame_skip", 2))
MAX_FPS = 15
# 训练模型路径：优先使用自定义训练结果，没有则回退到官方预训练
MODEL_PATH = CONFIG.get("model_path", "runs/segment/train/weights/best.pt")
FALLBACK_MODEL = CONFIG.get("fallback_model_path", "yolov8s-seg.pt")
# =======================================================

# ==========默认摄像头配置（运行时可通过后台动态新增视频源）==========
CAMERAS = [
    {"id": "cam1", "name": "配电间门口", "source": 0, "zone": [30, 30, 70, 70], "kind": "camera"},
    {"id": "cam2", "name": "实验室入口", "source": 1, "zone": [25, 35, 65, 75], "kind": "camera"},
    {"id": "cam3", "name": "楼道隔离区", "source": -1, "zone": [40, 25, 80, 70], "kind": "camera"},
    {"id": "cam4", "name": "施工区域", "source": -1, "zone": [20, 40, 60, 80], "kind": "camera"},
]

# 全局状态缓存（动态，video源运行时新增）
camera_states = {}
# 每个源的线程句柄
camera_threads = {}
# 需要停止的源集合（删除视频源时置入）
_camera_stop = set()

def _init_state(cid):
    if cid not in camera_states:
        camera_states[cid] = {
            "violation_count": 0,
            "alarm_level": "none",
            "records": [],
            "frame": None,
            "fps": 0,
            "connected": False,
            "last_time": time.time(),
            "frame_count": 0,
            "last_violation_start": None,
            "infer_counter": 0,
            "last_infer_result": None,
            "alarm_photo_taken": False,
            # ---- 危险会话聚合（需求：同人多次进入合并，记次数与总停留时长）----
            "danger_session_active": False,
            "danger_session_start": None,
            "danger_appearances": 0,
            "danger_episode_start": None,
            "danger_total_duration": 0.0,
            "danger_last_leave": None,
            "danger_alert_triggered": False,
            "danger_photo_taken": False,
            "danger_record_id": None,
            # 白名单进入是否已记录（本次在场期间只记一次）
            "whitelist_recorded": False,
        }
    return camera_states[cid]

def _should_stop(cid):
    return cid in _camera_stop

# 全局报警标志（任一摄像头red告警就持续响）
global_alarm_active = False

# 监控摄像头是否被暂停（供白名单拍照等使用，前端可查询状态）
monitor_paused = False

model = None

# ====== SQLite 数据库初始化 ======
DB_PATH = "monitor.db"

def _add_column_if_missing(c, table, column_def):
    """为已存在的表补充缺失的列（表结构迁移）"""
    cols = [row[1] for row in c.execute(f"PRAGMA table_info({table})").fetchall()]
    col_name = column_def.strip().split()[0]
    if col_name not in cols:
        c.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 用户表
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  created_time TEXT NOT NULL)''')
    # 告警记录表（含已处理/未处理状态字段）
    c.execute('''CREATE TABLE IF NOT EXISTS alarm_records
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  cam_id TEXT NOT NULL,
                  cam_name TEXT NOT NULL,
                  alarm_type TEXT NOT NULL,
                  duration INTEGER DEFAULT 0,
                  appearances INTEGER DEFAULT 1,
                  photo_path TEXT,
                  status TEXT DEFAULT 'unhandled',
                  timestamp TEXT NOT NULL)''')
    _add_column_if_missing(c, "alarm_records", "status TEXT DEFAULT 'unhandled'")
    _add_column_if_missing(c, "alarm_records", "appearances INTEGER DEFAULT 1")
    # 白名单人员表（存储人脸照片文件路径）
    c.execute('''CREATE TABLE IF NOT EXISTS whitelist
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  face_encoding_path TEXT NOT NULL,
                  created_time TEXT NOT NULL)''')
    conn.commit()
    conn.close()

def hash_password(pwd):
    return hashlib.sha256(pwd.encode('utf-8')).hexdigest()

def query_db(query, args=(), one=False):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(query, args)
    rv = cur.fetchall()
    conn.commit()
    conn.close()
    return (rv[0] if rv else None) if one else rv

init_db()

def load_model():
    global model
    if model is not None:
        return model
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 优先使用自定义训练模型
    chosen = MODEL_PATH if os.path.exists(MODEL_PATH) else FALLBACK_MODEL
    print(f"🚀 加载模型: {chosen} | 设备:{device}")
    model = YOLO(chosen)
    model.conf = CONF_THRESH
    model.imgsz = 640
    model.verbose = False
    if device == 'cuda':
        model.to('cuda')
    return model

def smooth_mask(mask_np):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)

def get_person_contour(mask, frame_shape, smooth_eps=2.2):
    if mask is None:
        return None
    h, w = frame_shape[:2]
    mask_np = mask.cpu().numpy().squeeze()
    mask_np = (mask_np > MASK_THRESH).astype(np.uint8) * 255
    mask_np = smooth_mask(mask_np)
    mask_resize = cv2.resize(mask_np, (w, h))
    contours, _ = cv2.findContours(mask_resize, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    max_contour = max(contours, key=cv2.contourArea)
    return cv2.approxPolyDP(max_contour, smooth_eps, closed=True)

def check_violation(contour, zone_pixel, box_xyxy=None):
    zx1, zy1, zx2, zy2 = zone_pixel
    if contour is not None and len(contour) >= 3:
        bottom_idx = contour[:, :, 1].argmax()
        foot_x, foot_y = contour[bottom_idx][0]
        mask_person = np.zeros((FIXED_H, FIXED_W), dtype=np.uint8)
        cv2.drawContours(mask_person, [contour], -1, 255, -1)
        mask_zone = np.zeros((FIXED_H, FIXED_W), dtype=np.uint8)
        cv2.rectangle(mask_zone, (zx1, zy1), (zx2, zy2), 255, -1)
        overlap = cv2.bitwise_and(mask_person, mask_zone)
        overlap_area = np.count_nonzero(overlap)
        person_area = np.count_nonzero(mask_person)
        overlap_ratio = overlap_area / person_area if person_area > 0 else 0
        foot_in = zx1 <= foot_x <= zx2 and zy1 <= foot_y <= zy2
        return overlap_ratio > 0.04 or foot_in, (int(foot_x), int(foot_y))
    if box_xyxy is not None:
        x1, y1, x2, y2 = box_xyxy
        foot_x = int((x1 + x2) / 2)
        foot_y = y2
        overlap = not (x2 < zx1 or x1 > zx2 or y2 < zy1 or y1 > zy2)
        foot_in = zx1 <= foot_x <= zx2 and zy1 <= foot_y <= zy2
        return overlap or foot_in, (foot_x, foot_y)
    return False, None

ALARM_PHOTO_DIR = os.path.join("static", "alarms")
os.makedirs(ALARM_PHOTO_DIR, exist_ok=True)

def save_alarm_photo(cam_id, frame):
    """保存告警照片（修复：确保目录存在，否则照片写不进去导致后台加载不出来）"""
    os.makedirs(ALARM_PHOTO_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"alarm_{cam_id}_{timestamp}.jpg"
    filepath = os.path.join(ALARM_PHOTO_DIR, filename)
    cv2.imwrite(filepath, frame)
    return f"/static/alarms/{filename}"

def add_alarm_record_to_db(cam_id, cam_name, alarm_type, duration, photo_path=None, appearances=1):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO alarm_records (cam_id, cam_name, alarm_type, duration, appearances, photo_path, status, timestamp) VALUES (?, ?, ?, ?, ?, ?, 'unhandled', ?)",
              (cam_id, cam_name, alarm_type, duration, appearances, photo_path, now))
    rid = c.lastrowid
    conn.commit()
    conn.close()
    return rid

def update_alarm_record(rid, duration=None, appearances=None, status=None):
    sets, args = [], []
    if duration is not None:
        sets.append("duration = ?"); args.append(int(duration))
    if appearances is not None:
        sets.append("appearances = ?"); args.append(int(appearances))
    if status is not None:
        sets.append("status = ?"); args.append(status)
    if sets:
        args.append(rid)
        query_db(f"UPDATE alarm_records SET {', '.join(sets)} WHERE id = ?", tuple(args))

def update_global_alarm_state():
    global global_alarm_active
    any_red = any(s["alarm_level"] == "red" for s in camera_states.values())
    if any_red and not global_alarm_active:
        global_alarm_active = True
        start_alarm()
    elif not any_red and global_alarm_active:
        global_alarm_active = False
        stop_alarm()

def _black_frame(cam_config, text):
    """生成无信号的黑色占位帧"""
    cid = cam_config["id"]
    state = camera_states[cid]
    zone_pct = cam_config["zone"]
    zx1 = int(zone_pct[0] / 100 * FIXED_W)
    zy1 = int(zone_pct[1] / 100 * FIXED_H)
    zx2 = int(zone_pct[2] / 100 * FIXED_W)
    zy2 = int(zone_pct[3] / 100 * FIXED_H)
    while not _should_stop(cid):
        frame = np.zeros((FIXED_H, FIXED_W, 3), dtype=np.uint8)
        cv2.putText(frame, text, (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2)
        cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 255, 255), 2)
        cv2.putText(frame, "ZONE", (zx1 + 4, zy1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(frame, cam_config["name"], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if state["alarm_level"] == "red":
            cv2.putText(frame, "RED ALARM", (FIXED_W - 130, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        elif state["alarm_level"] == "yellow":
            cv2.putText(frame, "WARNING", (FIXED_W - 120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "SAFE", (FIXED_W - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        state["frame"] = frame
        time.sleep(0.04)
    return

def _no_signal(cam_config):
    _black_frame(cam_config, "NO SIGNAL")

def _camera_fail(cam_config):
    _black_frame(cam_config, "CAMERA FAIL")

def _available_camera_indices(max_idx=6):
    """探测当前系统上真正可用的摄像头索引（解决外接摄像头后索引错位、第二路显示不了）"""
    valid = []
    for i in range(max_idx):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ok, _ = cap.read()
                cap.release()
                if ok:
                    valid.append(i)
        except Exception:
            continue
    return valid

def _auto_fix_camera_sources():
    """启动前把真实摄像头自动分配到可用索引，修复 cam2(source=1) 等配置对不上导致黑屏的问题"""
    valid = _available_camera_indices()
    if not valid:
        print("⚠️ 未探测到可用摄像头索引")
        return
    used = set()
    for cam in CAMERAS:
        if cam.get("kind") == "camera" and isinstance(cam["source"], int) and cam["source"] >= 0:
            if cam["source"] in valid and cam["source"] not in used:
                used.add(cam["source"])
            else:
                for v in valid:
                    if v not in used:
                        old = cam["source"]
                        cam["source"] = v
                        used.add(v)
                        print(f"📷 摄像头 {cam['id']}({cam['name']}) 索引自动修正: {old} → {v}")
                        break
    print(f"📷 可用摄像头索引: {valid}")

def _finalize_danger_session(st, cid, cam_name):
    """结算一次危险会话：写最终停留时长与出现次数，并复位会话状态"""
    if st["danger_record_id"] is not None:
        update_alarm_record(st["danger_record_id"],
                            duration=int(st["danger_total_duration"]),
                            appearances=st["danger_appearances"])
    st["danger_session_active"] = False
    st["danger_session_start"] = None
    st["danger_appearances"] = 0
    st["danger_total_duration"] = 0.0
    st["danger_last_leave"] = None
    st["danger_alert_triggered"] = False
    st["danger_photo_taken"] = False
    st["danger_record_id"] = None
    if st["alarm_level"] != "none":
        st["alarm_level"] = "none"

# ==================== 白名单人脸识别（OpenCV 标准 LBPH 识别器）====================
# 解决“设置白名单后进入警戒区仍被报警”：监控中识别到白名单人员时免报警、单独记录。
# 使用 cv2.face.LBPHFaceRecognizer（opencv-contrib），比自研直方图匹配更稳定。
_face_cascade = None
_LBPH = None            # cv2.face.LBPHFaceRecognizer
_WL_MAP = {}            # label(int) -> {"id": .., "name": ..}
_FACE_SIZE = (96, 96)
# LBPH 置信度阈值（越小越严格，匹配距离超过该值判为非白名单）
FACE_CONF_THRESH = float(CONFIG.get("face_confidence_threshold", 85))

def _get_face_cascade():
    global _face_cascade
    if _face_cascade is None:
        xml = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        _face_cascade = cv2.CascadeClassifier(xml)
    return _face_cascade

def _align_face(face_img):
    """归一化人脸：灰度 → 固定尺寸 → 直方图均衡（LBPH 输入）"""
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img
    gray = cv2.resize(gray, _FACE_SIZE)
    gray = cv2.equalizeHist(gray)
    return gray

def _detect_face_crop(img, min_size=40):
    """在图片中检测最大人脸并返回带适当边距的裁剪"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    faces = _get_face_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size))
    if len(faces) == 0:
        return None
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    m1, m2 = int(fw * 0.12), int(fh * 0.15)
    x0 = max(0, fx - m1); y0 = max(0, fy - m2)
    x1 = min(img.shape[1], fx + fw + m1); y1 = min(img.shape[0], fy + fh + m2)
    return img[y0:y1, x0:x1]

def refresh_whitelist_faces():
    """从数据库加载白名单人脸并用 LBPH 训练识别器（增删改白名单后调用刷新）"""
    global _LBPH, _WL_MAP
    _WL_MAP = {}
    crops, labels = [], []
    rows = query_db("SELECT id, name, face_encoding_path FROM whitelist")
    for r in rows:
        rel = r["face_encoding_path"].lstrip("/")
        if not os.path.exists(rel):
            continue
        img = cv2.imread(rel)
        if img is None:
            continue
        crop = _detect_face_crop(img)
        if crop is None or crop.size == 0:
            continue
        crops.append(_align_face(crop))
        lab = len(_WL_MAP)
        labels.append(lab)
        _WL_MAP[lab] = {"id": r["id"], "name": r["name"]}
    if crops:
        _LBPH = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=FACE_CONF_THRESH)
        _LBPH.train(crops, np.array(labels, dtype=np.int32))
    else:
        _LBPH = None
    print(f"🧑 白名单人脸库已加载: {len(_WL_MAP)} 人 (LBPH)")

def match_whitelist(face_crop):
    """LBPH 识别：返回 (白名单人员dict或None, 置信度)。置信度<=阈值判为命中。"""
    if _LBPH is None or not _WL_MAP:
        return None, None
    try:
        label, conf = _LBPH.predict(_align_face(face_crop))
    except Exception:
        return None, None
    if label in _WL_MAP and conf <= FACE_CONF_THRESH:
        return _WL_MAP[label], conf
    return None, conf

def _person_is_whitelisted(frame, box):
    """在人体框的上部区域检测人脸并与白名单比对；命中返回 (True, 姓名, 置信度)"""
    x1, y1, x2, y2 = box
    bh = y2 - y1
    if bh < 40:
        return False, None, None
    fy1 = max(0, y1)
    fy2 = min(frame.shape[0], y1 + int(bh * 0.45))
    fx1, fx2 = max(0, x1), min(frame.shape[1], x2)
    if fy2 - fy1 < 16 or fx2 - fx1 < 16:
        return False, None, None
    roi = frame[fy1:fy2, fx1:fx2]
    crop = _detect_face_crop(roi, min_size=24)
    if crop is None or crop.size == 0:
        return False, None, None
    w, conf = match_whitelist(crop)
    if w is not None:
        return True, w["name"], conf
    return False, None, conf

def _process_frame(cam_config, frame, is_video):
    """对单帧做警戒区检测，返回叠加了标注的帧。核心检测逻辑（含拍照/入库/报警）。"""
    cid = cam_config["id"]
    state = camera_states[cid]
    if is_video:
        frame = cv2.resize(frame, (FIXED_W, FIXED_H))
    h, w = FIXED_H, FIXED_W
    zone_pct = cam_config["zone"]
    zx1 = int(zone_pct[0] / 100 * w)
    zy1 = int(zone_pct[1] / 100 * h)
    zx2 = int(zone_pct[2] / 100 * w)
    zy2 = int(zone_pct[3] / 100 * h)
    zone_pixel = (zx1, zy1, zx2, zy2)

    # 绘制警戒区
    zone_overlay = frame.copy()
    cv2.rectangle(zone_overlay, (zx1, zy1), (zx2, zy2), (0, 255, 255), -1)
    cv2.addWeighted(zone_overlay, 0.15, frame, 0.85, 0, frame)
    cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 255, 255), 2)
    cv2.putText(frame, "ZONE", (zx1 + 4, zy1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # 跳帧推理
    state["infer_counter"] += 1
    if state["infer_counter"] >= FRAME_SKIP or state["last_infer_result"] is None:
        results = model(frame, retina_masks=True)
        state["last_infer_result"] = results[0]
        state["infer_counter"] = 0
    res = state["last_infer_result"]

    foot_inside_flag = False
    if res is not None and res.boxes is not None and len(res.boxes) > 0:
        for idx, box in enumerate(res.boxes):
            cls = int(box.cls[0].item())
            conf = box.conf[0].item()
            if cls != 0 or conf < CONF_THRESH:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            if (x2 - x1) * (y2 - y1) < AREA_THRESH:
                continue
            contour = None
            if res.masks is not None and idx < len(res.masks):
                try:
                    mask_tensor = res.masks[idx].data
                    contour = get_person_contour(mask_tensor, frame.shape)
                except Exception:
                    pass
            is_violate, foot_pt = check_violation(contour, zone_pixel, (x1, y1, x2, y2))
            if is_violate:
                # ===== 白名单免报警：识别到白名单人员则放行，不触发告警 =====
                wl_name = None
                try:
                    is_wl, wl_name, _ = _person_is_whitelisted(frame, (x1, y1, x2, y2))
                except Exception:
                    is_wl = False
                if is_wl:
                    # 白名单人员：放行不报警，但后台单独记录一条“白名单进入”（本次在场期间只记一次）
                    st = state
                    if not st["whitelist_recorded"]:
                        st["whitelist_recorded"] = True
                        photo_path = save_alarm_photo(cid, frame)
                        add_alarm_record_to_db(cid, cam_config['name'], 'whitelist', 0, photo_path, appearances=1)
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st["records"].append({"id": len(st["records"]) + 1, "time": now, "type": f"白名单进入:{wl_name} | {cam_config['name']}"})
                    foot_inside_flag = True
                    draw_color = (0, 255, 0)
                    if contour is not None and len(contour) >= 3:
                        cv2.drawContours(frame, [contour], -1, draw_color, 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
                    cv2.putText(frame, f"白名单:{wl_name}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, draw_color, 2)
                    continue
                # ===== 危险会话聚合：同人多次进入合并，记次数与总停留时长 =====
                foot_inside_flag = True
                now_t = time.time()
                st = state
                if st["danger_episode_start"] is None:
                    if not st["danger_session_active"]:
                        # 新会话（首次进入）
                        st["danger_session_active"] = True
                        st["danger_session_start"] = now_t
                        st["danger_appearances"] = 1
                        st["danger_total_duration"] = 0.0
                        st["danger_alert_triggered"] = False
                        st["danger_photo_taken"] = False
                        st["danger_record_id"] = None
                        if st["alarm_level"] != "yellow":
                            st["alarm_level"] = "yellow"
                            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st["records"].append({"id": len(st["records"]) + 1, "time": now, "type": f"人员进入警戒区 | {cam_config['name']}"})
                    else:
                        # 短时间再次进入：同一会话，累计出现次数
                        st["danger_appearances"] += 1
                    st["danger_episode_start"] = now_t
                live_dur = st["danger_total_duration"] + (now_t - st["danger_episode_start"])
                if live_dur >= ALARM_DURATION_THRESH:
                    draw_color = (0, 0, 255)
                    fill_alpha = 0.25
                    text = "OVERTIME ALARM"
                    if not st["danger_alert_triggered"]:
                        st["danger_alert_triggered"] = True
                        st["alarm_level"] = "red"
                        st["violation_count"] += 1
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st["records"].append({"id": len(st["records"]) + 1, "time": now, "type": f"超时越界 {int(live_dur)} 秒 | {cam_config['name']}"})
                        # 整个会话只拍一张照片、只记一条记录
                        photo_path = None
                        if not st["danger_photo_taken"]:
                            photo_path = save_alarm_photo(cid, frame)
                            st["danger_photo_taken"] = True
                        st["danger_record_id"] = add_alarm_record_to_db(cid, cam_config['name'], 'red', int(live_dur), photo_path, appearances=st["danger_appearances"])
                        update_global_alarm_state()
                    else:
                        # 已触发过告警：仅实时刷新记录时长/次数
                        if st["danger_record_id"] is not None:
                            update_alarm_record(st["danger_record_id"], duration=int(live_dur), appearances=st["danger_appearances"])
                else:
                    draw_color = (0, 255, 255)
                    fill_alpha = 0.22
                    text = "IN ZONE WARNING"
                # 绘制轮廓
                if contour is not None and len(contour) >= 3:
                    ov = frame.copy()
                    cv2.drawContours(ov, [contour], -1, draw_color, -1)
                    cv2.addWeighted(ov, fill_alpha, frame, 1 - fill_alpha, 0, frame)
                    cv2.drawContours(frame, [contour], -1, draw_color, 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
                cv2.circle(frame, foot_pt, 6, (0, 0, 255), -1)
                cv2.putText(frame, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)
                # 显示会话汇总（累计时长 + 出现次数）
                cv2.putText(frame, f"{int(live_dur)}s x{st['danger_appearances']}次", (x1, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            else:
                if contour is not None and len(contour) >= 3:
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 无人越界/无人：结算当前会话
    now_t = time.time()
    if not foot_inside_flag:
        st = state
        # 无人越界/无人：复位“白名单进入”标记（下次白名单人员再进入会重新记录）
        st["whitelist_recorded"] = False
        if st["danger_episode_start"] is not None:
            st["danger_total_duration"] += now_t - st["danger_episode_start"]
            st["danger_episode_start"] = None
            st["danger_last_leave"] = now_t
            if st["alarm_level"] == "red":
                st["alarm_level"] = "none"
        # 会话合并窗口到期 → 结算该会话（写最终时长/次数）
        if st["danger_session_active"] and st["danger_last_leave"] is not None and (now_t - st["danger_last_leave"]) > REENTRY_WINDOW:
            _finalize_danger_session(st, cid, cam_config['name'])
        update_global_alarm_state()

    cv2.putText(frame, cam_config["name"], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"C:{state['violation_count']}", (5, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    if state["alarm_level"] == "red":
        cv2.putText(frame, "RED ALARM", (w - 130, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    elif state["alarm_level"] == "yellow":
        cv2.putText(frame, "WARNING", (w - 120, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "SAFE", (w - 100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def camera_worker(cam_config):
    """统一视频源处理：摄像头(int) / 视频文件或URL(str) / 关闭(-1/'off')"""
    global global_alarm_active
    cid = cam_config["id"]
    src = cam_config["source"]
    _init_state(cid)
    state = camera_states[cid]

    is_video = isinstance(src, str)
    is_off = (isinstance(src, int) and src < 0) or src == "off" or src is None

    # 关闭/无信号
    if is_off:
        _no_signal(cam_config)
        return

    model_loaded = load_model()

    # 打开视频源（文件路径/URL字符串/摄像头索引）
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"打开源失败: {cam_config['name']}({src})，使用黑屏替代")
        _camera_fail(cam_config)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video else None
    frame_index = 0
    state["connected"] = True
    print(f"源启动成功: {cam_config['name']}({src})")

    while not _should_stop(cid):
        ret, raw = cap.read()
        if not ret:
            # 视频播放完毕则循环重播（方便演示）；摄像头失败则短暂等待
            if is_video:
                if total_frames and total_frames > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_index = 0
                    continue
                time.sleep(0.05)
                continue
            time.sleep(0.02)
            continue
        frame_index += 1
        if is_video and total_frames and frame_index >= total_frames:
            frame_index = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame = cv2.resize(raw, (FIXED_W, FIXED_H))
        # 仅真实摄像头做镜像翻转，视频源保持原样
        if not is_video:
            frame = cv2.flip(frame, 1)
        frame = _process_frame(cam_config, frame, is_video)
        state["frame"] = frame
        state["frame_count"] += 1
        if time.time() - state["last_time"] >= 1.0:
            state["fps"] = state["frame_count"]
            state["frame_count"] = 0
            state["last_time"] = time.time()

    cap.release()
    state["connected"] = False

# Flask 应用
app = Flask(__name__)
app.secret_key = 'your-secret-key-change-in-production'
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["VIDEO_FOLDER"] = "static/uploads/videos"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["VIDEO_FOLDER"], exist_ok=True)
os.makedirs(os.path.join("static", "faces"), exist_ok=True)
os.makedirs(os.path.join("static", "alarms"), exist_ok=True)

# 禁止浏览器缓存表单/自动回填账号密码（需求3：登录后第二次不要自动保存账号密码）
@app.after_request
def add_no_store_headers(resp):
    if request.path in ("/login", "/register", "/"):
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
    return resp

# ====== 登录校验装饰器 ======
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ====== 认证路由 ======
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = query_db("SELECT * FROM users WHERE username = ? AND password_hash = ?",
                        (username, hash_password(password)), one=True)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            return redirect(url_for('index'))
        return render_template('login.html', error='用户名或密码错误')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        confirm = request.form['confirm']
        if password != confirm:
            return render_template('register.html', error='两次密码不一致')
        if len(username) < 3 or len(password) < 6:
            return render_template('register.html', error='用户名至少3位，密码至少6位')
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            query_db("INSERT INTO users (username, password_hash, created_time) VALUES (?, ?, ?)",
                     (username, hash_password(password), now))
            return redirect(url_for('login'))
        except Exception:
            return render_template('register.html', error='用户名已存在')
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ====== 视频流 ======
def gen_stream(cam_id):
    while True:
        st = camera_states.get(cam_id)
        if st and st["frame"] is not None:
            ret, buf = cv2.imencode(".jpg", st["frame"], [cv2.IMWRITE_JPEG_QUALITY, 40])
            yield b"--frame\r\nContent-Type:image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        time.sleep(0.04)

@app.route("/video_feed/<cam_id>")
@login_required
def video_feed(cam_id):
    return Response(gen_stream(cam_id), mimetype="multipart/x-mixed-replace; boundary=frame")

# ====== 主页面 ======
@app.route("/")
@login_required
def index():
    return render_template("index.html", cameras=CAMERAS, username=session.get('username'))

@app.route("/api/cameras")
@login_required
def api_cameras():
    out = []
    for cam in CAMERAS:
        s = _init_state(cam["id"])
        out.append({
            "id": cam["id"],
            "name": cam["name"],
            "kind": cam.get("kind", "camera"),
            "violation_count": s["violation_count"],
            "alarm_level": s["alarm_level"],
            "fps": s["fps"],
            "connected": s["connected"],
            "records": s["records"][-20:],
            "zone": cam["zone"]
        })
    return jsonify(out)

@app.route("/api/set_zone/<cam_id>", methods=["POST"])
@login_required
def set_zone(cam_id):
    data = request.get_json()
    new_zone = [round(data["x1"]), round(data["y1"]), round(data["x2"]), round(data["y2"])]
    for c in CAMERAS:
        if c["id"] == cam_id:
            c["zone"] = new_zone
            break
    return jsonify({"status": "ok", "zone": new_zone})

@app.route("/api/clear_records", methods=["POST"])
@login_required
def clear_records():
    for c in CAMERAS:
        sid = c["id"]
        _init_state(sid)
        camera_states[sid]["records"] = []
        camera_states[sid]["violation_count"] = 0
        camera_states[sid]["alarm_level"] = "none"
        camera_states[sid]["last_violation_start"] = None
    stop_alarm()
    global global_alarm_active
    global_alarm_active = False
    return jsonify({"status": "ok"})

@app.route("/api/upload_map", methods=["POST"])
@login_required
def upload_map():
    file = request.files.get("map")
    if file:
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], "map.jpg")
        file.save(save_path)
        return jsonify({"status": "ok"})
    return jsonify({"error": "无文件"}), 400

# ====== 摄像头暂停/恢复（释放被占用的摄像头，供白名单拍照等使用）======
def _set_paused_frame(cam):
    """把摄像头画面替换为“已暂停”占位帧，并清空告警状态"""
    state = _init_state(cam["id"])
    frame = np.zeros((FIXED_H, FIXED_W, 3), dtype=np.uint8)
    cv2.putText(frame, "CAMERA PAUSED", (140, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
    cv2.putText(frame, "摄像头已暂停，可到白名单页拍照", (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 110, 110), 1)
    cv2.putText(frame, cam["name"], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    state["frame"] = frame
    state["connected"] = False
    state["alarm_level"] = "none"

def _is_real_webcam(cam):
    return cam.get("kind") == "camera" and isinstance(cam["source"], int) and cam["source"] >= 0

@app.route('/api/monitor/pause', methods=["POST"])
@login_required
def pause_cameras():
    """暂停所有真实摄像头（释放摄像头占用），视频源不受影响"""
    global monitor_paused
    for cam in CAMERAS:
        if _is_real_webcam(cam):
            _camera_stop.add(cam["id"])
            _set_paused_frame(cam)
    monitor_paused = True
    update_global_alarm_state()
    return jsonify({"status": "ok"})

@app.route('/api/monitor/resume', methods=["POST"])
@login_required
def resume_cameras():
    """恢复所有真实摄像头"""
    global monitor_paused
    for cam in CAMERAS:
        if _is_real_webcam(cam):
            _camera_stop.discard(cam["id"])
            st = _init_state(cam["id"])
            st["connected"] = False
            st["frame"] = None
            t = threading.Thread(target=camera_worker, args=(cam,), daemon=True)
            camera_threads[cam["id"]] = t
            t.start()
    monitor_paused = False
    return jsonify({"status": "ok"})

@app.route('/api/monitor/status', methods=["GET"])
@login_required
def monitor_status():
    """查询监控摄像头当前是否处于暂停状态（供白名单页等前端展示）"""
    return jsonify({"paused": monitor_paused})

# ====== 演示视频源管理（需求6：后台自动上传在线/本地视频做报警演示）======
def _sanitize_filename(name):
    name = re.sub(r'[\\/:*?"<>|\s]+', '_', name)
    return name[:60] or "video"

@app.route("/api/videos")
@login_required
def api_videos():
    vids = [c for c in CAMERAS if c.get("kind") == "video"]
    out = [{"id": c["id"], "name": c["name"], "source": c["source"], "zone": c["zone"]} for c in vids]
    return jsonify(out)

@app.route("/api/video/source", methods=["POST"])
@login_required
def add_video_source():
    video_url = (request.form.get("video_url") or "").strip()
    video_file = request.files.get("video_file")

    local_path = None
    if video_file and video_file.filename:
        ext = os.path.splitext(video_file.filename)[1] or ".mp4"
        fname = f"video_{int(time.time())}_{_sanitize_filename(os.path.splitext(video_file.filename)[0])}{ext}"
        local_path = os.path.join(app.config["VIDEO_FOLDER"], fname)
        video_file.save(local_path)
    elif video_url:
        # 下载在线视频到本地（更稳定），失败则尝试直接流式
        try:
            resp = requests.get(video_url, stream=True, timeout=30,
                                headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            ext = os.path.splitext(urllib.parse.urlparse(video_url).path)[1] or ".mp4"
            fname = f"video_{int(time.time())}_{_sanitize_filename(urllib.parse.urlparse(video_url).path.strip('/'))}{ext}"
            local_path = os.path.join(app.config["VIDEO_FOLDER"], fname)
            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
            print(f"已下载在线视频 -> {local_path}")
        except Exception as e:
            return jsonify({"error": f"在线视频下载失败: {e}"}), 400
    else:
        return jsonify({"error": "请提供视频URL或上传视频文件"}), 400

    if not local_path or not os.path.exists(local_path):
        return jsonify({"error": "视频保存失败"}), 400

    cid = f"video{int(time.time())}"
    cam = {"id": cid, "name": os.path.basename(local_path), "source": local_path,
           "zone": [30, 30, 70, 70], "kind": "video"}
    CAMERAS.append(cam)
    _init_state(cid)
    t = threading.Thread(target=camera_worker, args=(cam,), daemon=True)
    camera_threads[cid] = t
    t.start()
    return jsonify({"status": "ok", "id": cid, "name": cam["name"]})

@app.route("/api/video/delete/<cam_id>", methods=["POST"])
@login_required
def delete_video_source(cam_id):
    global global_alarm_active
    cam = next((c for c in CAMERAS if c["id"] == cam_id and c.get("kind") == "video"), None)
    if not cam:
        return jsonify({"error": "视频源不存在"}), 404
    _camera_stop.add(cam_id)
    CAMERAS.remove(cam)
    # 尝试删除本地文件
    try:
        if isinstance(cam["source"], str) and os.path.exists(cam["source"]):
            os.remove(cam["source"])
    except Exception:
        pass
    st = camera_states.get(cam_id)
    if st:
        st["alarm_level"] = "none"
        st["connected"] = False
    update_global_alarm_state()
    return jsonify({"status": "ok"})

# ====== 告警记录页面（需求4：照片修复 + 已处理/未处理 + 删除）======
@app.route("/records")
@login_required
def records_page():
    return render_template("records.html", username=session.get('username'))

@app.route("/api/alarms")
@login_required
def api_alarms():
    status = request.args.get("status", "all")
    ftype = request.args.get("type", "all")
    conds, args = [], []
    if status in ("unhandled", "handled"):
        conds.append("status = ?")
        args.append(status)
    if ftype == "whitelist":
        conds.append("alarm_type = 'whitelist'")
    elif ftype == "alarm":
        conds.append("alarm_type != 'whitelist'")
    where = ("WHERE " + " AND ".join(conds)) if conds else ""
    rows = query_db(f"SELECT * FROM alarm_records {where} ORDER BY id DESC", tuple(args))
    out = [{
        "id": r["id"],
        "cam_name": r["cam_name"],
        "alarm_type": r["alarm_type"],
        "duration": r["duration"],
        "appearances": r["appearances"],
        "photo_path": r["photo_path"],
        "status": r["status"],
        "timestamp": r["timestamp"]
    } for r in rows]
    return jsonify(out)

@app.route("/api/alarms/<int:rid>/status", methods=["POST"])
@login_required
def set_alarm_status(rid):
    data = request.get_json() or {}
    status = data.get("status", "handled")
    if status not in ("handled", "unhandled"):
        return jsonify({"error": "状态非法"}), 400
    query_db("UPDATE alarm_records SET status = ? WHERE id = ?", (status, rid))
    return jsonify({"status": "ok"})

@app.route("/api/alarms/<int:rid>/delete", methods=["POST"])
@login_required
def delete_alarm(rid):
    row = query_db("SELECT photo_path FROM alarm_records WHERE id = ?", (rid,), one=True)
    if row:
        # 删除关联的照片文件
        if row["photo_path"]:
            rel = row["photo_path"].lstrip("/")
            if os.path.exists(rel):
                try:
                    os.remove(rel)
                except Exception:
                    pass
        query_db("DELETE FROM alarm_records WHERE id = ?", (rid,))
        return jsonify({"status": "ok"})
    return jsonify({"error": "记录不存在"}), 404

@app.route("/api/alarms/clear", methods=["POST"])
@login_required
def clear_alarms():
    """一键清空全部告警记录（删除所有记录及其现场照片文件）"""
    rows = query_db("SELECT photo_path FROM alarm_records")
    for r in rows:
        if r["photo_path"]:
            rel = r["photo_path"].lstrip("/")
            if os.path.exists(rel):
                try:
                    os.remove(rel)
                except Exception:
                    pass
    query_db("DELETE FROM alarm_records")
    return jsonify({"status": "ok"})

# ====== 白名单人脸管理（需求1：摄像头拍照/选择照片两种录入；需求2：查看/重命名/删除）======
@app.route("/whitelist")
@login_required
def whitelist_page():
    people = query_db("SELECT * FROM whitelist ORDER BY id DESC")
    return render_template("whitelist.html", people=people, username=session.get('username'))

@app.route("/api/whitelist/add", methods=["POST"])
@login_required
def add_whitelist():
    name = (request.form.get("name") or "").strip()
    file = request.files.get("face_image")
    if not name or not file:
        return jsonify({"error": "姓名和照片必填"}), 400
    # 摄像头拍照(blob)与选择照片(file)走同一接口
    filename = f"face_{int(time.time())}_{_sanitize_filename(name)}.jpg"
    save_path = os.path.join("static", "faces", filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file.save(save_path)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    query_db("INSERT INTO whitelist (name, face_encoding_path, created_time) VALUES (?, ?, ?)",
             (name, f"/static/faces/{filename}", now))
    refresh_whitelist_faces()
    return jsonify({"status": "ok"})

@app.route("/api/whitelist/rename/<int:pid>", methods=["POST"])
@login_required
def rename_whitelist(pid):
    data = request.get_json() or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "姓名不能为空"}), 400
    query_db("UPDATE whitelist SET name = ? WHERE id = ?", (name, pid))
    refresh_whitelist_faces()
    return jsonify({"status": "ok"})

@app.route("/api/whitelist/delete/<int:pid>", methods=["POST"])
@login_required
def delete_whitelist(pid):
    row = query_db("SELECT face_encoding_path FROM whitelist WHERE id = ?", (pid,), one=True)
    if row and row["face_encoding_path"]:
        rel = row["face_encoding_path"].lstrip("/")
        if os.path.exists(rel):
            try:
                os.remove(rel)
            except Exception:
                pass
    query_db("DELETE FROM whitelist WHERE id = ?", (pid,))
    refresh_whitelist_faces()
    return jsonify({"status": "ok"})

@app.route("/api/whitelist/clear", methods=["POST"])
@login_required
def clear_whitelist():
    """一键清空全部白名单（删除记录与人脸照片文件）"""
    rows = query_db("SELECT face_encoding_path FROM whitelist")
    for r in rows:
        if r["face_encoding_path"]:
            rel = r["face_encoding_path"].lstrip("/")
            if os.path.exists(rel):
                try:
                    os.remove(rel)
                except Exception:
                    pass
    query_db("DELETE FROM whitelist")
    refresh_whitelist_faces()
    return jsonify({"status": "ok"})

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    print("=" * 60)
    print("  越界监测系统 v2")
    print("  功能：登录注册 / 白名单管理(摄像头拍照/选照片) / 告警记录(已处理/未处理/删除)")
    print("        / 演示视频源(在线视频/本地视频) / 自定义数据集训练模型加载")
    print("  默认账号：admin / admin123 (请先注册自己的账号)")
    print("  本地地址：http://127.0.0.1:5000")
    print("=" * 60)

    existing = query_db("SELECT * FROM users WHERE username = ?", ("admin",), one=True)
    if not existing:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        query_db("INSERT INTO users (username, password_hash, created_time) VALUES (?, ?, ?)",
                 ("admin", hash_password("admin123"), now))
        print("已创建默认管理员账号: admin / admin123")

    webbrowser.open("http://127.0.0.1:5000/login")
    load_model()
    refresh_whitelist_faces()
    _auto_fix_camera_sources()
    for cam in CAMERAS:
        t = threading.Thread(target=camera_worker, args=(cam,), daemon=True)
        camera_threads[cam["id"]] = t
        t.start()
        time.sleep(0.3)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
