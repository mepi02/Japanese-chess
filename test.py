import cv2
import numpy as np
from joblib import load

# HOG設定（学習時と同じパラメータで）
hog = cv2.HOGDescriptor(
    _winSize=(64,64),
    _blockSize=(16,16),
    _blockStride=(8,8),
    _cellSize=(8,8),
    _nbins=9
)

# モデルとラベルエンコーダの読み込み
clf, le = load('shogi_model.joblib')

# 画像の読み込みと前処理
test_img_path = 'Original/kinn.png'  # 認識したい画像ファイル名に変更
img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f'画像が読み込めません: {test_img_path}')
img = cv2.resize(img, (64,64))
_, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
descriptor = hog.compute(img)

# 推論と信頼度
if hasattr(clf, 'predict_proba'):
    proba = clf.predict_proba([descriptor.flatten()])[0]
    pred = proba.argmax()
    label = le.inverse_transform([pred])[0]
    confidence = proba[pred]
    print(f"認識結果: {label} (信頼度: {confidence:.2f})")
else:
    pred = clf.predict([descriptor.flatten()])
    label = le.inverse_transform(pred)[0]
    print(f"認識結果: {label} (信頼度: 不明)")