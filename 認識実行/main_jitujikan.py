
import cv2
import mediapipe as mp
import numpy as np
import math
import pickle
import time
import matplotlib.pyplot as plt
pi = math.pi

# landmarkの繋がり表示用
landmark_line_ids = [
    (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 0),  # 掌
    (1, 2), (2, 3), (3, 4),         # 親指
    (5, 6), (6, 7), (7, 8),         # 人差し指
    (9, 10), (10, 11), (11, 12),    # 中指
    (13, 14), (14, 15), (15, 16),   # 薬指
    (17, 18), (18, 19), (19, 20),   # 小指
]

num = 0
count2 = 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,                    # 最大検出数
    min_detection_confidence=0.7,       # 検出信頼度
    min_tracking_confidence=0.7         # 追跡信頼度
)
count_1 = count_2 = count_3 = count_4 = count_5 = count_6 = count_7 = count_8 = count_9 = count_10 = count_11 = count_12 = count_13 = count_14 = count_15 = count_16 = count_17 = count_18 = count_19 = count_20 = count_21 = count_22 = count_23 = count_24 = count_25 = count_26 = count_27 = count_28 = count_29 = count_30 = count_31 = count_32 = count_33 = count_34 = count_35 = count_36 = count_37 = count_38 = count_39 = count_40 = count_41 = 0
count_hand = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
hand_moji = ['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ', 'さ', 'し', 'す', 'せ', 'そ', 'た', 'ち', 'つ', 'て', 'と', 'な', 'に', 'ぬ', 'ね', 'は', 'ひ', 'ふ', 'へ', 'ほ', 'ま', 'み', 'む', 'め', 'や', 'ゆ', 'よ', 'ら', 'る', 'れ', 'ろ', 'わ']
with open('hand.pickle', mode='rb') as f:
    svc = pickle.load(f)
src = cv2.imread("入力.png")
src_1 = cv2.imread("あ.png")
src_2 = cv2.imread("い.png")
src_3 = cv2.imread("う.png")
src_4 = cv2.imread("え.png")
src_5 = cv2.imread("お.png")
src_6 = cv2.imread("か.png")
src_7 = cv2.imread("き.png")
src_8 = cv2.imread("く.png")
src_9 = cv2.imread("け.png")
src_10 = cv2.imread("こ.png")
src_11 = cv2.imread("さ.png")
src_12 = cv2.imread("し.png")
src_13 = cv2.imread("す.png")
src_14 = cv2.imread("せ.png")
src_15 = cv2.imread("そ.png")
src_16 = cv2.imread("た.png")
src_17 = cv2.imread("ち.png")
src_18 = cv2.imread("つ.png")
src_19 = cv2.imread("て.png")
src_20 = cv2.imread("と.png")
src_21 = cv2.imread("な.png")
src_22 = cv2.imread("に.png")
src_23 = cv2.imread("ぬ.png")
src_24 = cv2.imread("ね.png")
src_25 = cv2.imread("は.png")
src_26 = cv2.imread("ひ.png")
src_27 = cv2.imread("ふ.png")
src_28 = cv2.imread("へ.png")
src_29 = cv2.imread("ほ.png")
src_30 = cv2.imread("ま.png")
src_31 = cv2.imread("み.png")
src_32 = cv2.imread("む.png")
src_33 = cv2.imread("め.png")
src_34 = cv2.imread("や.png")
src_35 = cv2.imread("ゆ.png")
src_36 = cv2.imread("よ.png")
src_37 = cv2.imread("ら.png")
src_38 = cv2.imread("る.png")
src_39 = cv2.imread("れ.png")
src_40 = cv2.imread("ろ.png")
src_41 = cv2.imread("わ.png")

cap = cv2.VideoCapture(0)
if cap.isOpened():
    while True:
        # カメラから画像を取得
        success, img = cap.read()
        if not success:
            continue

        img = cv2.flip(img, 1)              # 左右を反転
        img_h, img_w, _ = img.shape         # サイズ取得
        # 検出処理の実行
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

                JUDGE_THUMB_1 = ((WRIST_x - THUMB_IP_x) * (WRIST_x - THUMB_IP_x)) + (
                        (WRIST_y - THUMB_IP_y) * (WRIST_y - THUMB_IP_y))  # - 親指
                JUDGE_THUMB_2 = ((WRIST_x - THUMB_TIP_x) * (WRIST_x - THUMB_TIP_x)) + (
                        (WRIST_y - THUMB_TIP_y) * (WRIST_y - THUMB_TIP_y))

                JUDGE_INDEX_FINGER_1 = ((WRIST_x - INDEX_FINGER_PIP_x) * (WRIST_x - INDEX_FINGER_PIP_x)) + (
                        (WRIST_y - INDEX_FINGER_PIP_y) * (WRIST_y - INDEX_FINGER_PIP_y))  # - 人差し指
                JUDGE_INDEX_FINGER_2 = ((WRIST_x - INDEX_FINGER_TIP_x) * (WRIST_x - INDEX_FINGER_TIP_x)) + (
                        (WRIST_y - INDEX_FINGER_TIP_y) * (WRIST_y - INDEX_FINGER_TIP_y))

                JUDGE_MIDDLE_FINGER_1 = ((WRIST_x - MIDDLE_FINGER_PIP_x) * (WRIST_x - MIDDLE_FINGER_PIP_x)) + (
                        (WRIST_y - MIDDLE_FINGER_PIP_y) * (WRIST_y - MIDDLE_FINGER_PIP_y))  # - 中指
                JUDGE_MIDDLE_FINGER_2 = ((WRIST_x - MIDDLE_FINGER_TIP_x) * (WRIST_x - MIDDLE_FINGER_TIP_x)) + (
                        (WRIST_y - MIDDLE_FINGER_TIP_y) * (WRIST_y - MIDDLE_FINGER_TIP_y))

                JUDGE_RING_FINGER_1 = ((WRIST_x - RING_FINGER_PIP_x) * (WRIST_x - RING_FINGER_PIP_x)) + (
                        (WRIST_y - RING_FINGER_PIP_y) * (WRIST_y - RING_FINGER_PIP_y))  # - 薬指
                JUDGE_RING_FINGER_2 = ((WRIST_x - RING_FINGER_TIP_x) * (WRIST_x - RING_FINGER_TIP_x)) + (
                        (WRIST_y - RING_FINGER_TIP_y) * (WRIST_y - RING_FINGER_TIP_y))

                JUDGE_PINKY_1 = ((WRIST_x - PINKY_PIP_x) * (WRIST_x - PINKY_PIP_x)) + (
                        (WRIST_y - PINKY_PIP_y) * (WRIST_y - PINKY_PIP_y))  # - 小指
                JUDGE_PINKY_2 = ((WRIST_x - PINKY_TIP_x) * (WRIST_x - PINKY_TIP_x)) + (
                        (WRIST_y - PINKY_TIP_y) * (WRIST_y - PINKY_TIP_y))

                JUDGE_THUMB_INDEX = ((THUMB_TIP_x - INDEX_FINGER_TIP_x) * (THUMB_TIP_x - INDEX_FINGER_TIP_x)) + (
                        (THUMB_TIP_y - INDEX_FINGER_TIP_y) * (THUMB_TIP_y - INDEX_FINGER_TIP_y))  # - 親指と人差し指の長さ
                JUDGE_THUMB_MIDDLE = ((THUMB_TIP_x - MIDDLE_FINGER_TIP_x) * (THUMB_TIP_x - MIDDLE_FINGER_TIP_x)) + (
                        (THUMB_TIP_y - MIDDLE_FINGER_TIP_y) * (THUMB_TIP_y - MIDDLE_FINGER_TIP_y))  # - 親指と中指の長さ
                JUDGE_THUMB_RING = ((THUMB_TIP_x - RING_FINGER_TIP_x) * (THUMB_TIP_x - RING_FINGER_TIP_x)) + (
                        (THUMB_TIP_y - RING_FINGER_TIP_y) * (THUMB_TIP_y - RING_FINGER_TIP_y))  # - 親指と薬指の長さ
                JUDGE_THUMB_PINKY = ((THUMB_TIP_x - PINKY_TIP_x) * (THUMB_TIP_x - PINKY_TIP_x)) + (
                        (THUMB_TIP_y - PINKY_TIP_y) * (THUMB_TIP_y - PINKY_TIP_y))  # - 親指と小指の長さ
                # 隣の指同士の距離
                NEXT_INDEX_MIDDLE = ((INDEX_FINGER_TIP_x - MIDDLE_FINGER_TIP_x) * (
                            INDEX_FINGER_TIP_x - MIDDLE_FINGER_TIP_x)) + (
                                            (INDEX_FINGER_TIP_y - MIDDLE_FINGER_TIP_y) * (
                                                INDEX_FINGER_TIP_y - MIDDLE_FINGER_TIP_y))
                NEXT_MIDDLE_RING = ((MIDDLE_FINGER_TIP_x - RING_FINGER_TIP_x) * (
                            MIDDLE_FINGER_TIP_x - RING_FINGER_TIP_x)) + (
                                           (MIDDLE_FINGER_TIP_y - RING_FINGER_TIP_y) * (
                                               MIDDLE_FINGER_TIP_y - RING_FINGER_TIP_y))
                NEXT_RING_PINKY = ((RING_FINGER_TIP_x - PINKY_TIP_x) * (RING_FINGER_TIP_x - PINKY_TIP_x)) + (
                        (RING_FINGER_TIP_y - PINKY_TIP_y) * (RING_FINGER_TIP_y - PINKY_TIP_y))

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
                        lm_xy_6 = hand_landmarks.landmark[1]
                        lm_xy_7 = hand_landmarks.landmark[10]
                        xy_1 = np.arctan2(lm_xy_1.x * img_w - lm_xy.x * img_w, lm_xy_1.y * img_h - lm_xy.y * img_h)
                        xy_2 = np.arctan2(lm_xy_2.x * img_w - lm_xy.x * img_w, lm_xy_2.y * img_h - lm_xy.y * img_h)
                        xy_3 = np.arctan2(lm_xy_3.x * img_w - lm_xy.x * img_w, lm_xy_3.y * img_h - lm_xy.y * img_h)
                        xy_4 = np.arctan2(lm_xy_4.x * img_w - lm_xy.x * img_w, lm_xy_4.y * img_h - lm_xy.y * img_h)
                        xy_5 = np.arctan2(lm_xy_5.x * img_w - lm_xy.x * img_w, lm_xy_5.y * img_h - lm_xy.y * img_h)
                        xy_6 = np.arctan2(lm_xy_1.x * img_w - lm_xy_6.x * img_w, lm_xy_1.y * img_h - lm_xy_6.y * img_h)
                        xy_7 = np.arctan2(lm_xy_3.x * img_w - lm_xy_7.x * img_w, lm_xy_3.y * img_h - lm_xy_7.y * img_h)
                        a = np.arctan2(lm_xy.x * img_w - lm.x * img_w, lm_xy.y * img_h - lm.y * img_h)
                        b = np.arctan2(lm_xy_1.x * img_w - lm.x * img_w, lm_xy_1.z * img_w - lm.z * img_w)
                        c = np.arctan2(lm_xy_5.x * img_w - lm.x * img_w, lm_xy_5.z * img_w - lm.z * img_w)
                        font = cv2.FONT_HERSHEY_SIMPLEX

                    # # - テキスト出力
                    # for cnt, text in enumerate(hand_texts1):
                    #     cv2.putText(img, text, (lm_x, lm_y + 10 * cnt), font, 0.7, lm_c, 1)

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
                    JUDGE_PINKY_1 / ReferencePoint_5,
                    (JUDGE_PINKY_1 / ReferencePoint_5) - (JUDGE_PINKY_2 / ReferencePoint_5),
                    JUDGE_THUMB_INDEX / ReferencePoint_Y,
                    JUDGE_THUMB_MIDDLE / ReferencePoint_Y, JUDGE_THUMB_RING / ReferencePoint_Y,
                    JUDGE_THUMB_PINKY / ReferencePoint_Y, NEXT_INDEX_MIDDLE / ReferencePoint_Y,
                    NEXT_MIDDLE_RING / ReferencePoint_Y, NEXT_RING_PINKY / ReferencePoint_Y, (180 * a) / pi,
                     (180 * b) / pi, (180 * c) / pi, (180 * xy_1) / pi, (180 * xy_2) / pi, (180 * xy_3) / pi,
                     (180 * xy_4) / pi, (180 * xy_5) / pi, (180 * xy_6) / pi, (180 * xy_7) / pi]
                ]
                ans = svc.predict(hand_data)

                print((ans))

                if ans == 'あ':
                    cv2.imshow("answer", src_1)
                elif ans == 'い':
                    cv2.imshow("answer", src_2)
                elif ans == 'う':
                    cv2.imshow("answer", src_3)
                elif ans == 'え':
                    cv2.imshow("answer", src_4)
                elif ans == 'お':
                    cv2.imshow("answer", src_5)
                elif ans == 'か':
                    cv2.imshow("answer", src_6)
                elif ans == 'き':
                    cv2.imshow("answer", src_7)
                elif ans == 'く':
                    cv2.imshow("answer", src_8)
                elif ans == 'け':
                    cv2.imshow("answer", src_9)
                elif ans == 'こ':
                    cv2.imshow("answer", src_10)
                elif ans == 'さ':
                    cv2.imshow("answer", src_11)
                elif ans == 'し':
                    cv2.imshow("answer", src_12)
                elif ans == 'す':
                    cv2.imshow("answer", src_13)
                elif ans == 'せ':
                    cv2.imshow("answer", src_14)
                elif ans == 'そ':
                    cv2.imshow("answer", src_15)
                elif ans == 'た':
                    cv2.imshow("answer", src_16)
                elif ans == 'ち':
                    cv2.imshow("answer", src_17)
                elif ans == 'つ':
                    cv2.imshow("answer", src_18)
                elif ans == 'て':
                    cv2.imshow("answer", src_19)
                elif ans == 'と':
                    cv2.imshow("answer", src_20)
                elif ans == 'な':
                    cv2.imshow("answer", src_21)
                elif ans == 'に':
                    cv2.imshow("answer", src_22)
                elif ans == 'ぬ':
                    cv2.imshow("answer", src_23)
                elif ans == 'ね':
                    cv2.imshow("answer", src_24)
                elif ans == 'は':
                    cv2.imshow("answer", src_25)
                elif ans == 'ひ':
                    cv2.imshow("answer", src_26)
                elif ans == 'ふ':
                    cv2.imshow("answer", src_27)
                elif ans == 'へ':
                    cv2.imshow("answer", src_28)
                elif ans == 'ほ':
                    cv2.imshow("answer", src_29)
                elif ans == 'ま':
                    cv2.imshow("answer", src_30)
                elif ans == 'み':
                    cv2.imshow("answer", src_31)
                elif ans == 'む':
                    cv2.imshow("answer", src_32)
                elif ans == 'め':
                    cv2.imshow("answer", src_33)
                elif ans == 'や':
                    cv2.imshow("answer", src_34)
                elif ans == 'ゆ':
                    cv2.imshow("answer", src_35)
                elif ans == 'よ':
                    cv2.imshow("answer", src_36)
                elif ans == 'ら':
                    cv2.imshow("answer", src_37)
                elif ans == 'る':
                    cv2.imshow("answer", src_38)
                elif ans == 'れ':
                    cv2.imshow("answer", src_39)
                elif ans == 'ろ':
                    cv2.imshow("answer", src_40)
                elif ans == 'わ':
                    cv2.imshow("answer", src_41)

        #  画像の表示
        cv2.imshow("MediaPipe Hands", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 0x1b:
            break

cap.release()
