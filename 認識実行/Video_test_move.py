import os
import cv2
import mediapipe as mp
import numpy as np
import math
import csv
import pickle
pi = math.pi
# landmarkの繋がり表示用
count_hand = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

landmark_line_ids = [
        (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 0),  # 掌
        (1, 2), (2, 3), (3, 4),         # 親指
        (5, 6), (6, 7), (7, 8),         # 人差し指
        (9, 10), (10, 11), (11, 12),    # 中指
        (13, 14), (14, 15), (15, 16),   # 薬指
        (17, 18), (18, 19), (19, 20),   # 小指
]
dakuten_hiragana = [
        "か", "き", "く", "け", "こ", "さ", "し", "す", "せ", "そ",
        "た", "ち", "つ", "て", "と", "は", "ひ", "ふ", "へ", "ほ"
        ]
hand_moji = ['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ', 'さ', 'し', 'す', 'せ', 'そ', 'た', 'ち', 'つ', 'て', 'と', 'な', 'に', 'ぬ', 'ね', 'は', 'ひ', 'ふ', 'へ', 'ほ', 'ま', 'み', 'む', 'め', 'や', 'ゆ', 'よ', 'ら', 'る', 'れ', 'ろ', 'わ']
input_characters = []
num = 0
count2 = 0
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
        max_num_hands=1,                    # 最大検出数
        min_detection_confidence=0.7,       # 検出信頼度
        min_tracking_confidence=0.7         # 追跡信頼度
)
# with open('hand_move.pickle', mode='rb') as f:
#         svc = pickle.load(f)

from collections import deque

class TextTracker:
    def __init__(self, min_continuous=15, start_frame=5, movement_threshold=5):
        self.min_continuous = min_continuous
        self.start_frame = start_frame
        self.movement_threshold = movement_threshold
        self.current_text = None
        self.continuous_count = 0
        self.total_movement_x = 0
        self.total_movement_y = 0
        self.reference_points_history = deque(maxlen=start_frame)

    def update(self, new_text, reference_point_x, reference_point_y):
        if self.current_text is None:
            self.current_text = new_text

        if new_text == self.current_text:
            self.continuous_count += 1
            if self.continuous_count >= self.start_frame:
                self.accumulate_movement(reference_point_x, reference_point_y)
        else:
            # 文字が変わったら初期化
            self.reset_tracker(new_text)

        if len(self.reference_points_history) == self.reference_points_history.maxlen:
            self.reference_points_history.popleft()
        self.reference_points_history.append((reference_point_x, reference_point_y))

        if self.continuous_count >= self.min_continuous:
            direction, distance = self.determine_major_movement()
            if abs(distance) >= self.movement_threshold:
                return self.current_text, direction, distance
        return None, None, None

    def accumulate_movement(self, current_x, current_y):
        last_x, last_y = self.reference_points_history[0]
        self.total_movement_x += current_x - last_x
        self.total_movement_y += current_y - last_y

    def determine_major_movement(self):
        if abs(self.total_movement_x) > abs(self.total_movement_y):
            return 'Right', abs(self.total_movement_x)
        else:
            return 'Up', abs(self.total_movement_y)

    def reset_tracker(self, new_text):
        self.current_text = new_text
        self.continuous_count = 0
        self.total_movement_x = 0
        self.total_movement_y = 0
        self.reference_points_history.clear()

with open('hand.pickle', mode='rb') as f:
        svc = pickle.load(f)
no = 0 # 画像の枚数
count_no = 0
save_dir = "./exfinger"
# os.mkdir(save_dir)  # 保存ディレクトリーを作成
# 動画ファイルから入力を開始
video_dir = "Video"
video_extensions = ('.mp4', '.mov', '.MOV', '.avi', '.AVI')

# Videoディレクトリ内の全動画ファイルを取得
video_files = []
for file in os.listdir(video_dir):
    if file.endswith(video_extensions):
        video_files.append(os.path.join(video_dir, file))

# 各動画ファイルを処理
for video_path in sorted(video_files):
    print(f"処理中: {video_path}")
    
    # 各動画ごとにTrackerを初期化
    tracker = TextTracker(min_continuous=15, start_frame=10, movement_threshold=100)
    
    cap = cv2.VideoCapture(video_path)
    
    if cap.isOpened():
        while True:
                success, img = cap.read()
                if not success:
                        print("ファイルなし\n")
                        break
                img = cv2.flip(img, 1)  # 左右を反転
                img_h, img_w, _ = img.shape  # サイズ取得
                results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if results.multi_hand_landmarks:
                        count = 0
                        # 検出した手の数分繰り返し
                        for h_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        # landmarkのつながりをlineで表示
                                # - それぞれの指の定義
                                WRIST = hand_landmarks.landmark[0]  # - 手のひら
                                WRIST_x = int(WRIST.x * img_w)
                                WRIST_y = int(WRIST.y * img_h)
                                WRIST_z = int(WRIST.z * img_w)
                                # 基準点
                                THUMB_MCP = hand_landmarks.landmark[1]
                                THUMB_MCP_x = int(THUMB_MCP.x * img_w)
                                THUMB_MCP_y = int(THUMB_MCP.y * img_h)

                                INDEX_FINGER_MCP = hand_landmarks.landmark[5]
                                INDEX_FINGER_MCP_x = int(INDEX_FINGER_MCP.x * img_w)
                                INDEX_FINGER_MCP_y = int(INDEX_FINGER_MCP.y * img_h)

                                MIDDLE_FINGER_MCP = hand_landmarks.landmark[9]
                                MIDDLE_FINGER_MCP_x = int(MIDDLE_FINGER_MCP.x * img_w)
                                MIDDLE_FINGER_MCP_y = int(MIDDLE_FINGER_MCP.y * img_h)

                                RING_FINGER_MCP = hand_landmarks.landmark[13]
                                RING_FINGER_MCP_x = int(RING_FINGER_MCP.x * img_w)
                                RING_FINGER_MCP_y = int(RING_FINGER_MCP.y * img_h)

                                PINKY_MCP = hand_landmarks.landmark[17]
                                PINKY_MCP_x = int(PINKY_MCP.x * img_w)
                                PINKY_MCP_y = int(PINKY_MCP.y * img_h)
                                # 相対座標
                                ReferencePoint_x = MIDDLE_FINGER_MCP_x - WRIST_x
                                ReferencePoint_y = MIDDLE_FINGER_MCP_y - WRIST_y
                                # 親指基準点
                                ReferencePoint_1 = ((WRIST_x - THUMB_MCP_x) * (WRIST_x - THUMB_MCP_x)) + (
                                        (WRIST_y - THUMB_MCP_y) * (WRIST_y - THUMB_MCP_y))
                                # 人差し指基準点
                                ReferencePoint_2 = ((WRIST_x - INDEX_FINGER_MCP_x) * (WRIST_x - INDEX_FINGER_MCP_x)) + (
                                        (WRIST_y - INDEX_FINGER_MCP_y) * (WRIST_y - INDEX_FINGER_MCP_y))
                                # 中指基準点
                                ReferencePoint_3 = ((WRIST_x - MIDDLE_FINGER_MCP_x) * (WRIST_x - MIDDLE_FINGER_MCP_x)) + (
                                        (WRIST_y - MIDDLE_FINGER_MCP_y) * (WRIST_y - MIDDLE_FINGER_MCP_y))
                                # 薬指基準点
                                ReferencePoint_4 = ((WRIST_x - RING_FINGER_MCP_x) * (WRIST_x - RING_FINGER_MCP_x)) + (
                                        (WRIST_y - RING_FINGER_MCP_y) * (WRIST_y - RING_FINGER_MCP_y))
                                # 小指基準点
                                ReferencePoint_5 = ((WRIST_x - INDEX_FINGER_MCP_x) * (WRIST_x - INDEX_FINGER_MCP_x)) + (
                                        (WRIST_y - PINKY_MCP_y) * (WRIST_y - PINKY_MCP_y))
                                ReferencePoint_Y = ((INDEX_FINGER_MCP_x - PINKY_MCP_x) * (INDEX_FINGER_MCP_x - PINKY_MCP_x)) + (
                                        (INDEX_FINGER_MCP_y - PINKY_MCP_y) * (INDEX_FINGER_MCP_y - PINKY_MCP_y))

                                THUMB_IP = hand_landmarks.landmark[3]  # - 親指
                                THUMB_IP_x = int(THUMB_IP.x * img_w)
                                THUMB_IP_y = int(THUMB_IP.y * img_h)
                                THUMB_TIP = hand_landmarks.landmark[4]
                                THUMB_TIP_x = int(THUMB_TIP.x * img_w)
                                THUMB_TIP_y = int(THUMB_TIP.y * img_h)
                                THUMB_TIP_z = int(THUMB_TIP.z)

                                INDEX_FINGER_PIP = hand_landmarks.landmark[7]  # - 人差し指
                                INDEX_FINGER_PIP_x = int(INDEX_FINGER_PIP.x * img_w)
                                INDEX_FINGER_PIP_y = int(INDEX_FINGER_PIP.y * img_h)
                                INDEX_FINGER_TIP = hand_landmarks.landmark[8]
                                INDEX_FINGER_TIP_x = int(INDEX_FINGER_TIP.x * img_w)
                                INDEX_FINGER_TIP_y = int(INDEX_FINGER_TIP.y * img_h)
                                INDEX_FINGER_TIP_z = int(INDEX_FINGER_TIP.z)

                                MIDDLE_FINGER_PIP = hand_landmarks.landmark[11]  # - 中指
                                MIDDLE_FINGER_PIP_x = int(MIDDLE_FINGER_PIP.x * img_w)
                                MIDDLE_FINGER_PIP_y = int(MIDDLE_FINGER_PIP.y * img_h)
                                MIDDLE_FINGER_TIP = hand_landmarks.landmark[12]
                                MIDDLE_FINGER_TIP_x = int(MIDDLE_FINGER_TIP.x * img_w)
                                MIDDLE_FINGER_TIP_y = int(MIDDLE_FINGER_TIP.y * img_h)
                                MIDDLE_FINGER_TIP_z = int(MIDDLE_FINGER_TIP.z)

                                RING_FINGER_PIP = hand_landmarks.landmark[15]  # - 薬指
                                RING_FINGER_PIP_x = int(RING_FINGER_PIP.x * img_w)
                                RING_FINGER_PIP_y = int(RING_FINGER_PIP.y * img_h)
                                RING_FINGER_TIP = hand_landmarks.landmark[16]
                                RING_FINGER_TIP_x = int(RING_FINGER_TIP.x * img_w)
                                RING_FINGER_TIP_y = int(RING_FINGER_TIP.y * img_h)
                                RING_FINGER_TIP_z = int(RING_FINGER_TIP.z)

                                PINKY_PIP = hand_landmarks.landmark[19]  # - 小指
                                PINKY_PIP_x = int(PINKY_PIP.x * img_w)
                                PINKY_PIP_y = int(PINKY_PIP.y * img_h)
                                PINKY_TIP = hand_landmarks.landmark[20]
                                PINKY_TIP_x = int(PINKY_TIP.x * img_w)
                                PINKY_TIP_y = int(PINKY_TIP.y * img_h)
                                PINKY_TIP_z = int(PINKY_TIP.z)
                                # 基準点からそれぞれの指の距離
                                # 親指
                                JUDGE_THUMB_1 = ((WRIST_x - THUMB_IP_x) * (WRIST_x - THUMB_IP_x)) + (
                                        (WRIST_y - THUMB_IP_y) * (WRIST_y - THUMB_IP_y))
                                JUDGE_THUMB_2 = ((WRIST_x - THUMB_TIP_x) * (WRIST_x - THUMB_TIP_x)) + (
                                        (WRIST_y - THUMB_TIP_y) * (WRIST_y - THUMB_TIP_y))
                                # 人差し指
                                JUDGE_INDEX_FINGER_1 = ((WRIST_x - INDEX_FINGER_PIP_x) * (WRIST_x - INDEX_FINGER_PIP_x)) + (
                                        (WRIST_y - INDEX_FINGER_PIP_y) * (WRIST_y - INDEX_FINGER_PIP_y))
                                JUDGE_INDEX_FINGER_2 = ((WRIST_x - INDEX_FINGER_TIP_x) * (WRIST_x - INDEX_FINGER_TIP_x)) + (
                                        (WRIST_y - INDEX_FINGER_TIP_y) * (WRIST_y - INDEX_FINGER_TIP_y))
                                # 中指
                                JUDGE_MIDDLE_FINGER_1 = ((WRIST_x - MIDDLE_FINGER_PIP_x) * (WRIST_x - MIDDLE_FINGER_PIP_x)) + (
                                        (WRIST_y - MIDDLE_FINGER_PIP_y) * (WRIST_y - MIDDLE_FINGER_PIP_y))
                                JUDGE_MIDDLE_FINGER_2 = ((WRIST_x - MIDDLE_FINGER_TIP_x) * (WRIST_x - MIDDLE_FINGER_TIP_x)) + (
                                        (WRIST_y - MIDDLE_FINGER_TIP_y) * (WRIST_y - MIDDLE_FINGER_TIP_y))
                                # 薬指
                                JUDGE_RING_FINGER_1 = ((WRIST_x - RING_FINGER_PIP_x) * (WRIST_x - RING_FINGER_PIP_x)) + (
                                        (WRIST_y - RING_FINGER_PIP_y) * (WRIST_y - RING_FINGER_PIP_y))
                                JUDGE_RING_FINGER_2 = ((WRIST_x - RING_FINGER_TIP_x) * (WRIST_x - RING_FINGER_TIP_x)) + (
                                        (WRIST_y - RING_FINGER_TIP_y) * (WRIST_y - RING_FINGER_TIP_y))
                                # 小指
                                JUDGE_PINKY_1 = ((WRIST_x - PINKY_PIP_x) * (WRIST_x - PINKY_PIP_x)) + (
                                        (WRIST_y - PINKY_PIP_y) * (WRIST_y - PINKY_PIP_y))
                                JUDGE_PINKY_2 = ((WRIST_x - PINKY_TIP_x) * (WRIST_x - PINKY_TIP_x)) + (
                                        (WRIST_y - PINKY_TIP_y) * (WRIST_y - PINKY_TIP_y))
                                # 親指からそれぞれの指の長さ
                                JUDGE_THUMB_INDEX = ((THUMB_TIP_x - INDEX_FINGER_TIP_x) * (THUMB_TIP_x - INDEX_FINGER_TIP_x)) + (
                                        (THUMB_TIP_y - INDEX_FINGER_TIP_y) * (THUMB_TIP_y - INDEX_FINGER_TIP_y))  # - 親指と人差し指の長さ
                                JUDGE_THUMB_MIDDLE = ((THUMB_TIP_x - MIDDLE_FINGER_TIP_x) * (THUMB_TIP_x - MIDDLE_FINGER_TIP_x)) + (
                                        (THUMB_TIP_y - MIDDLE_FINGER_TIP_y) * (THUMB_TIP_y - MIDDLE_FINGER_TIP_y))  # - 親指と中指の長さ
                                JUDGE_THUMB_RING = ((THUMB_TIP_x - RING_FINGER_TIP_x) * (THUMB_TIP_x - RING_FINGER_TIP_x)) + (
                                        (THUMB_TIP_y - RING_FINGER_TIP_y) * (THUMB_TIP_y - RING_FINGER_TIP_y))  # - 親指と薬指の長さ
                                JUDGE_THUMB_PINKY = ((THUMB_TIP_x - PINKY_TIP_x) * (THUMB_TIP_x - PINKY_TIP_x)) + (
                                        (THUMB_TIP_y - PINKY_TIP_y) * (THUMB_TIP_y - PINKY_TIP_y))  # - 親指と小指の長さ
                                # 親指の第一関節からそれぞれの指の長さ new
                                JUDGE_THUMB_2_INDEX = ((THUMB_IP_x - INDEX_FINGER_TIP_x) * (THUMB_IP_x - INDEX_FINGER_TIP_x)) + (
                                        (THUMB_IP_y - INDEX_FINGER_TIP_y) * (THUMB_IP_y - INDEX_FINGER_TIP_y))  # - 親指と人差し指の長さ
                                JUDGE_THUMB_2_MIDDLE = ((THUMB_IP_x - MIDDLE_FINGER_TIP_x) * (THUMB_IP_x - MIDDLE_FINGER_TIP_x)) + (
                                        (THUMB_IP_y - MIDDLE_FINGER_TIP_y) * (THUMB_IP_y - MIDDLE_FINGER_TIP_y))  # - 親指と中指の長さ
                                JUDGE_THUMB_2_RING = ((THUMB_IP_x - RING_FINGER_TIP_x) * (THUMB_IP_x - RING_FINGER_TIP_x)) + (
                                        (THUMB_IP_y - RING_FINGER_TIP_y) * (THUMB_IP_y - RING_FINGER_TIP_y))  # - 親指と薬指の長さ
                                JUDGE_THUMB_2_PINKY = ((THUMB_IP_x - PINKY_TIP_x) * (THUMB_IP_x - PINKY_TIP_x)) + (
                                        (THUMB_IP_y - PINKY_TIP_y) * (THUMB_IP_y - PINKY_TIP_y))  # - 親指と小指の長さ
                                # 親指からそれぞれの第一関節の長さ new
                                JUDGE_THUMB_INDEX_2 = ((THUMB_TIP_x - INDEX_FINGER_PIP_x) * (THUMB_TIP_x - INDEX_FINGER_PIP_x)) + (
                                        (THUMB_TIP_y - INDEX_FINGER_PIP_y) * (THUMB_TIP_y - INDEX_FINGER_PIP_y))  # - 親指と人差し指の長さ
                                JUDGE_THUMB_MIDDLE_2 = ((THUMB_TIP_x - MIDDLE_FINGER_PIP_x) * (THUMB_TIP_x - MIDDLE_FINGER_PIP_x)) + (
                                        (THUMB_TIP_y - MIDDLE_FINGER_PIP_y) * (THUMB_TIP_y - MIDDLE_FINGER_PIP_y))  # - 親指と中指の長さ
                                JUDGE_THUMB_RING_2 = ((THUMB_TIP_x - RING_FINGER_PIP_x) * (THUMB_TIP_x - RING_FINGER_PIP_x)) + (
                                        (THUMB_TIP_y - RING_FINGER_PIP_y) * (THUMB_TIP_y - RING_FINGER_PIP_y))  # - 親指と薬指の長さ
                                JUDGE_THUMB_PINKY_2 = ((THUMB_TIP_x - PINKY_PIP_x) * (THUMB_TIP_x - PINKY_PIP_x)) + (
                                        (THUMB_TIP_y - PINKY_PIP_y) * (THUMB_TIP_y - PINKY_PIP_y))  # - 親指と小指の長さ
                                # 親指の第一関節からそれぞれの第一関節の長さ new
                                JUDGE_THUMB_2_INDEX_2 = ((THUMB_IP_x - INDEX_FINGER_PIP_x) * (THUMB_IP_x - INDEX_FINGER_PIP_x)) + (
                                        (THUMB_IP_y - INDEX_FINGER_PIP_y) * (THUMB_IP_y - INDEX_FINGER_PIP_y))  # - 親指と人差し指の長さ
                                JUDGE_THUMB_2_MIDDLE_2 = ((THUMB_IP_x - MIDDLE_FINGER_PIP_x) * (THUMB_IP_x - MIDDLE_FINGER_PIP_x)) + (
                                        (THUMB_IP_y - MIDDLE_FINGER_PIP_y) * (THUMB_IP_y - MIDDLE_FINGER_PIP_y))  # - 親指と中指の長さ
                                JUDGE_THUMB_2_RING_2 = ((THUMB_IP_x - RING_FINGER_PIP_x) * (THUMB_IP_x - RING_FINGER_PIP_x)) + (
                                        (THUMB_IP_y - RING_FINGER_PIP_y) * (THUMB_IP_y - RING_FINGER_PIP_y))  # - 親指と薬指の長さ
                                JUDGE_THUMB_2_PINKY_2 = ((THUMB_IP_x - PINKY_PIP_x) * (THUMB_IP_x - PINKY_PIP_x)) + (
                                        (THUMB_IP_y - PINKY_PIP_y) * (THUMB_IP_y - PINKY_PIP_y))  # - 親指と小指の長さ
                                # 人差し指からそれぞれの指の先
                                JUDGE_INDEX_MIDDLE = ((INDEX_FINGER_TIP_x - MIDDLE_FINGER_TIP_x) * (INDEX_FINGER_TIP_x - MIDDLE_FINGER_TIP_x)) + (
                                        (INDEX_FINGER_TIP_y - MIDDLE_FINGER_TIP_y) * (INDEX_FINGER_TIP_y - MIDDLE_FINGER_TIP_y))  # - 親指と中指の長さ
                                JUDGE_INDEX_RING = ((INDEX_FINGER_TIP_x - RING_FINGER_TIP_x) * (INDEX_FINGER_TIP_x - RING_FINGER_TIP_x)) + (
                                        (INDEX_FINGER_TIP_y - RING_FINGER_TIP_y) * (INDEX_FINGER_TIP_y - RING_FINGER_TIP_y))  # - 親指と薬指の長さ
                                JUDGE_INDEX_PINKY = ((INDEX_FINGER_TIP_x - PINKY_TIP_x) * (INDEX_FINGER_TIP_x - PINKY_TIP_x)) + (
                                        (INDEX_FINGER_TIP_y - PINKY_TIP_y) * (INDEX_FINGER_TIP_y - PINKY_TIP_y))  # - 親指と小指の長さ
                                # 中指からそれぞれの距離
                                JUDGE_MIDDLE_RING = ((INDEX_FINGER_TIP_x - RING_FINGER_TIP_x) * (INDEX_FINGER_TIP_x - RING_FINGER_TIP_x)) + (
                                        (INDEX_FINGER_TIP_y - RING_FINGER_TIP_y) * (INDEX_FINGER_TIP_y - RING_FINGER_TIP_y))  # - 親指と薬指の長さ
                                JUDGE_MIDDLE_PINKY = ((INDEX_FINGER_TIP_x - PINKY_TIP_x) * (INDEX_FINGER_TIP_x - PINKY_TIP_x)) + (
                                        (INDEX_FINGER_TIP_y - PINKY_TIP_y) * (INDEX_FINGER_TIP_y - PINKY_TIP_y))  # - 親指と小指の長さ
                                # 隣の指同士の距離
                                NEXT_INDEX_MIDDLE = ((INDEX_FINGER_TIP_x - MIDDLE_FINGER_TIP_x) * (
                                        INDEX_FINGER_TIP_x - MIDDLE_FINGER_TIP_x)) + (
                                        (INDEX_FINGER_TIP_y - MIDDLE_FINGER_TIP_y) * (INDEX_FINGER_TIP_y - MIDDLE_FINGER_TIP_y))
                                NEXT_MIDDLE_RING = ((MIDDLE_FINGER_TIP_x - RING_FINGER_TIP_x) * (
                                        MIDDLE_FINGER_TIP_x - RING_FINGER_TIP_x)) + (
                                        (MIDDLE_FINGER_TIP_y - RING_FINGER_TIP_y) * (MIDDLE_FINGER_TIP_y - RING_FINGER_TIP_y))
                                NEXT_RING_PINKY = ((RING_FINGER_TIP_x - PINKY_TIP_x) * (RING_FINGER_TIP_x - PINKY_TIP_x)) + (
                                        (RING_FINGER_TIP_y - PINKY_TIP_y) * (RING_FINGER_TIP_y - PINKY_TIP_y))
                                # 基準点からz座標の距離
                                Z_THUMB = ((WRIST_x - THUMB_TIP_x) * (WRIST_x - THUMB_TIP_x)) + (
                                        (WRIST_z - THUMB_TIP_z) * (WRIST_z - THUMB_TIP_z))
                                Z_INDEX = ((WRIST_x - INDEX_FINGER_TIP_x) * (WRIST_x - INDEX_FINGER_TIP_x)) + (
                                        (WRIST_z - INDEX_FINGER_TIP_z) * (WRIST_z - INDEX_FINGER_TIP_z))
                                Z_MIDDLE = ((WRIST_x - MIDDLE_FINGER_TIP_x) * (WRIST_x - MIDDLE_FINGER_TIP_x)) + (
                                        (WRIST_z - MIDDLE_FINGER_TIP_z) * (WRIST_z - MIDDLE_FINGER_TIP_z))
                                Z_RING = ((WRIST_x - RING_FINGER_TIP_x) * (WRIST_x - RING_FINGER_TIP_x)) + (
                                        (WRIST_z - RING_FINGER_TIP_z) * (WRIST_z - RING_FINGER_TIP_z))
                                Z_PINKY = ((WRIST_x - PINKY_TIP_x) * (WRIST_x - PINKY_TIP_x)) + (
                                        (WRIST_z - PINKY_TIP_z) * (WRIST_z - PINKY_TIP_z))

                                for line_id in landmark_line_ids:
                                        # 1点目の座標取得
                                        lm = hand_landmarks.landmark[line_id[0]]
                                        lm_pos1 = (int(lm.x * img_w), int(lm.y * img_h))

                                        if lm == hand_landmarks.landmark[3] or lm == hand_landmarks.landmark[4] or lm == \
                                                hand_landmarks.landmark[7] or lm == hand_landmarks.landmark[8] or lm == \
                                                hand_landmarks.landmark[11] or lm == hand_landmarks.landmark[12] or lm == \
                                                hand_landmarks.landmark[15] or lm == hand_landmarks.landmark[16] or lm == \
                                                hand_landmarks.landmark[19] or lm == hand_landmarks.landmark[20]:
                                                        cv2.circle(img, lm_pos1, 10, (0, 0, 255), thickness=-1)
                                        else:
                                                cv2.circle(img, lm_pos1, 7, (255, 255, 255), thickness=-1)

                                        # 2点目の座標取得
                                        lm = hand_landmarks.landmark[line_id[1]]
                                        lm_pos2 = (int(lm.x * img_w), int(lm.y * img_h))
                                        if lm == hand_landmarks.landmark[3] or lm == hand_landmarks.landmark[4] or lm == \
                                                hand_landmarks.landmark[7] or lm == hand_landmarks.landmark[8] or lm == \
                                                hand_landmarks.landmark[11] or lm == hand_landmarks.landmark[12] or lm == \
                                                hand_landmarks.landmark[15] or lm == hand_landmarks.landmark[16] or lm == \
                                                hand_landmarks.landmark[19] or lm == hand_landmarks.landmark[20]:
                                                cv2.circle(img, lm_pos2, 10, (0, 0, 255), thickness=-1)
                                        else:
                                                cv2.circle(img, lm_pos2, 7, (255, 255, 255), thickness=-1)

                                        cv2.line(img, lm_pos1, lm_pos2, (255, 255, 255), 3)

                                        hand_texts1 = []  # - 座標表示
                                        for c_id, hand_class in enumerate(results.multi_handedness[h_id].classification):
                                                # - テキスト表示に必要な座標など準備
                                                lm = hand_landmarks.landmark[0]
                                                lm_x = int(lm.x * img_w) - 120
                                                lm_y = int(lm.y * img_h) + 30
                                                lm_c = (64, 0, 0)
                                                X1 = (int(lm.x * img_w), int(lm.y * img_h))
                                                lm_xy = hand_landmarks.landmark[9]  # 基準点(角度用)
                                                lm_xy_1 = hand_landmarks.landmark[4]
                                                lm_xy_2 = hand_landmarks.landmark[8]
                                                lm_xy_3 = hand_landmarks.landmark[12]
                                                lm_xy_4 = hand_landmarks.landmark[16]
                                                lm_xy_5 = hand_landmarks.landmark[20]
                                                lm_xy_6 = hand_landmarks.landmark[3]
                                                lm_xy_7 = hand_landmarks.landmark[10]
                                                lm_xy_8 = hand_landmarks.landmark[18]
                                                lm_xy_9 = hand_landmarks.landmark[6]
                                                lm_xy_10 = hand_landmarks.landmark[14]
                                                xy_1 = np.arctan2(lm_xy_1.x * img_w - lm_xy.x * img_w, lm_xy_1.y * img_h - lm_xy.y * img_h)
                                                xy_2 = np.arctan2(lm_xy_2.x * img_w - lm_xy.x * img_w, lm_xy_2.y * img_h - lm_xy.y * img_h)
                                                xy_3 = np.arctan2(lm_xy_3.x * img_w - lm_xy.x * img_w, lm_xy_3.y * img_h - lm_xy.y * img_h)
                                                xy_4 = np.arctan2(lm_xy_4.x * img_w - lm_xy.x * img_w, lm_xy_4.y * img_h - lm_xy.y * img_h)
                                                xy_5 = np.arctan2(lm_xy_5.x * img_w - lm_xy.x * img_w, lm_xy_5.y * img_h - lm_xy.y * img_h)
                                                xy_6 = np.arctan2(lm_xy_1.x * img_w - lm_xy_6.x * img_w, lm_xy_1.y * img_h - lm_xy_6.y * img_h)
                                                xy_7 = np.arctan2(lm_xy_3.x * img_w - lm_xy_7.x * img_w, lm_xy_3.y * img_h - lm_xy_7.y * img_h)
                                                xy_8 = np.arctan2(lm_xy_5.x * img_w - lm_xy_8.x * img_w, lm_xy_5.y * img_h - lm_xy_8.y * img_h)
                                                xy_9 = np.arctan2(lm_xy_2.x * img_w - lm_xy_9.x * img_w, lm_xy_2.y * img_h - lm_xy_9.y * img_h)
                                                xy_10 = np.arctan2(lm_xy_4.x * img_w - lm_xy_10.x * img_w, lm_xy_4.y * img_h - lm_xy_10.y * img_h)

                                                a = np.arctan2(lm_xy.x * img_w - lm.x * img_w, lm_xy.y * img_h - lm.y * img_h)
                                                b = np.arctan2(lm_xy_1.x * img_w - lm.x * img_w, lm_xy_1.z * img_w - lm.z * img_w)
                                                c = np.arctan2(lm_xy_5.x * img_w - lm.x * img_w, lm_xy_5.z * img_w - lm.z * img_w)
                                                font = cv2.FONT_HERSHEY_SIMPLEX
                                num += 1
                                hand_data = [
                                        [JUDGE_THUMB_1 / ReferencePoint_1,
                                        (JUDGE_THUMB_1 / ReferencePoint_1) - (JUDGE_THUMB_2 / ReferencePoint_1),
                                        JUDGE_INDEX_FINGER_1 / ReferencePoint_2,
                                        (JUDGE_INDEX_FINGER_1 / ReferencePoint_2) - (JUDGE_INDEX_FINGER_2 / ReferencePoint_2),
                                        JUDGE_MIDDLE_FINGER_1 / ReferencePoint_3,
                                        (JUDGE_MIDDLE_FINGER_1 / ReferencePoint_3) - (JUDGE_MIDDLE_FINGER_2 / ReferencePoint_3),
                                        JUDGE_RING_FINGER_1 / ReferencePoint_4,
                                        (JUDGE_RING_FINGER_1 / ReferencePoint_4) - (JUDGE_RING_FINGER_2 / ReferencePoint_4),
                                        JUDGE_PINKY_1 / ReferencePoint_5 * 2,
                                        ((JUDGE_PINKY_1 / ReferencePoint_5) - (JUDGE_PINKY_2 / ReferencePoint_5)) * 2,
                                        JUDGE_THUMB_INDEX / ReferencePoint_Y,
                                        JUDGE_THUMB_MIDDLE / ReferencePoint_Y, JUDGE_THUMB_RING / ReferencePoint_Y,
                                        JUDGE_THUMB_PINKY / ReferencePoint_Y, NEXT_INDEX_MIDDLE / ReferencePoint_Y,
                                        NEXT_MIDDLE_RING / ReferencePoint_Y, NEXT_RING_PINKY / ReferencePoint_Y,
                                        a, b, c, xy_1, xy_2, xy_3, xy_4, xy_5, xy_6, xy_7, xy_8, xy_9, xy_10]
                                ]
                                ans = svc.predict(hand_data)
                                # ans_move = svc.predict(hand_data_move)
                                # outfile = save_dir + "/" + str(no) + str(ans) + ".jpg"
                                # cv2.imwrite(outfile, img)
                                # no += 1
                                # 追跡を更新するために、認識された文字と参照点の座標を渡します。
                                result, direction, distance = tracker.update(new_text=ans, reference_point_x=ReferencePoint_x, reference_point_y=ReferencePoint_y)

                                # print(f"Result: {result}, Direction: {direction}, Distance: {distance}")
                                print(ans)
                                f = open('Finger_data_move.csv', mode="a", newline="")
                                writer = csv.writer(f)

                                # Assuming hand_data is a list of data points
                                for data in hand_data:
                                        writer.writerow([result, direction, distance])  # Enclose your data in a list
                                f.close()

    cap.release()
    cv2.destroyAllWindows()
    print(f"完了: {video_path}")

print("全ての動画の処理が完了しました")