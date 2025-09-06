import cv2
import numpy as np
import random

def augment(img):
    h, w = img.shape[:2]

    # --- 1. 小回転（±5〜10度） ---
    angle = random.uniform(-10, 10)  # -10〜10度
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # --- 2. 明るさ変化 ---
    value = random.randint(-40, 40)  # -40〜40の範囲で加減
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = cv2.split(hsv)
    v_ = np.clip(v_.astype(np.int16) + value, 0, 255).astype(np.uint8)
    hsv = cv2.merge((h_, s_, v_))
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # --- 3. 少しスケール変化 ---
    scale = random.uniform(0.9, 1.1)  # 90〜110%
    M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # --- 4. ガウシアンノイズ追加 ---
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)  # 平均0, 標準偏差10
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


    return img
for i in range(1, 101):
    img = cv2.imread("Original/ryuuou.png")   # カラー画像
    augmented = augment(img)
    cv2.imwrite(f"data/ryuuou_up/ryuuou_up_0{i}.png", augmented)
    rotated = cv2.rotate(augmented, cv2.ROTATE_180)
    cv2.imwrite(f"data/ryuuou_down/ryuuou_down_0{i}.png", rotated)