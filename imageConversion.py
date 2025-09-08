import cv2
import numpy as np
import random

def augment(img):
    h, w = img.shape[:2]

    # --- 1. 小回転（±10度） ---
    if random.random() < 0.7:
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # --- 2. 明るさ変化 ---
    if random.random() < 0.7:
        value = random.randint(-50, 50)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_, s_, v_ = cv2.split(hsv)
        v_ = np.clip(v_.astype(np.int16) + value, 0, 255).astype(np.uint8)
        hsv = cv2.merge((h_, s_, v_))
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # --- 3. スケール変化 ---
    if random.random() < 0.7:
        scale = random.uniform(0.8, 1.2)
        M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # --- 4. 平行移動 ---
    if random.random() < 0.7:
        tx, ty = random.randint(-7,7), random.randint(-7,7)
        M = np.float32([[1,0,tx],[0,1,ty]])
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # --- 5. コントラスト変化 ---
    if random.random() < 0.7:
        alpha = random.uniform(0.5, 1.5)  # コントラスト
        beta = random.randint(-30, 30)    # 明るさオフセット
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # --- 6. ぼかし ---
    if random.random() < 0.3:
        img = cv2.GaussianBlur(img, (1,1), 0)

    # --- 7. 部分隠し (Cutout) ---
    if random.random() < 0.1:
        x, y = random.randint(0, w-15), random.randint(0, h-15)
        cw, ch = random.randint(5,15), random.randint(5,15)
        img[y:y+ch, x:x+cw] = 0

    # --- 8. ガウシアンノイズ追加 ---
    if random.random() < 0.3:
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img
for i in range(1, 601):
    img = cv2.imread("Original/ginn.png")   # カラー画像
    augmented = augment(img)
    cv2.imwrite(f"data/ginn_up/ginn_up_{i:03}.png", augmented)
    rotated = cv2.rotate(augmented, cv2.ROTATE_180)
    cv2.imwrite(f"data/ginn_down/ginn_down_{i:03}.png", rotated)