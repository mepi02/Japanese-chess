from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import cv2
import numpy as np
import glob
from joblib import dump

# HOG
hog = cv2.HOGDescriptor(
    _winSize=(64,64),
    _blockSize=(16,16),
    _blockStride=(8,8),
    _cellSize=(8,8),
    _nbins=9
)

X = []
y = []

labels = ['fu_up','fu_down',
          'ginn_up','ginn_down',
          'gyoku_up','gyoku_down',
          'hisya_up','hisya_down',
          'kaku_up','kaku_down',
          'keima_up','keima_down',
          'kin_up','kin_down',
          'kousya_up','kousya_down',
          'nariginn_up','nariginn_down',
          'narikei_up','narikei_down',
          'narikou_up','narikou_down',
          'oogoma_up','oogoma_down',
          'ou_up','ou_down',
          'ryuuou_up','ryuuou_down',
          'tokinn_up','tokinn_down',
          'None']

# データ読み込み
for label in labels:
    for file in glob.glob(f'data/{label}/*.png'):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        # 64x64 にリサイズ
        img = cv2.resize(img, (64,64))
        # 前処理（例: 二値化）
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        descriptor = hog.compute(img)
        X.append(descriptor.flatten())
        y.append(label)

X = np.array(X)
y = np.array(y)

# ラベルを整数化
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# 訓練・テスト分割
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)



# SVM学習（信頼度出力のためprobability=True）
clf = svm.SVC(kernel='rbf', C=10, probability=True)
clf.fit(X_train, y_train)
dump(clf, 'shogi_model.joblib')


# 評価
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# モデル保存
dump((clf, le), 'shogi_model.joblib')