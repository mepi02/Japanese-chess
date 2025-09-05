import cv2
import numpy as np

# 画像読み込み
image = cv2.imread("test2.png")
image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# ノイズ軽減
image_blur = cv2.GaussianBlur(image_gray, (5,5), 0)

# Cannyエッジ
edges = cv2.Canny(image_blur, 30, 100)

# 線を太らせる
kernel = np.ones((3,3), np.uint8)
edges = cv2.dilate(edges, kernel, iterations=2)

# 輪郭検出
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

image_result = image.copy()

# 四角形数をカウント
square_count = 0

# 白背景画像（描画用）
image_blank = np.ones_like(image) * 255

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 5000 or area>20000:  # 小さい輪郭を除外
        cv2.drawContours(image_result, [cnt], -1, (255,0,0), 2),
        continue

    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

    if len(approx) == 4:
        square_count += 1
        cv2.drawContours(image_result, [approx], -1, (0,255,0), 2)
    else:
        cv2.drawContours(image_result, [approx], -1, (0,0,255), 2)
    

# 結果保存
cv2.imwrite('detected_squares.png', image_result)

# 結果出力
print(f"検出された四角形の数: {square_count}")