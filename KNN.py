import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib



# Reading data
BASE_DIR       = r"../data"
TRAIN_IMG_DIR  = os.path.join(BASE_DIR, "train")
VAL_IMG_DIR    = os.path.join(BASE_DIR, "validation")
TRAIN_CSV      = os.path.join(BASE_DIR, "train.csv")
VAL_CSV        = os.path.join(BASE_DIR, "validation.csv")

# Hyperparameters
BINS       = (8, 8, 8)   # 512‑D
NEIGHBORS  = 5
SEED       = 42
np.random.seed(SEED)

# Extracting the histogram (TF and NumPy)
def rgb_histogram_tf(path, bins=BINS):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    flat = tf.reshape(img, [-1, 3]).numpy().astype(np.float32)  # (N,3)
    hist, _ = np.histogramdd(flat,
                             bins=bins,
                             range=[[0,255],[0,255],[0,255]])
    hist = hist.flatten()
    return hist / (hist.sum() + 1e-7)

# Loading CSVs
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)
train_df["file_path"] = train_df["image_id"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, f"{x}.png"))
val_df["file_path"]   = val_df["image_id"].apply(lambda x: os.path.join(VAL_IMG_DIR,   f"{x}.png"))

# Feature matrix
print("[k‑NN] extracting histograms with TensorFlow …")
train_X = np.stack([rgb_histogram_tf(p) for p in train_df.file_path])
val_X   = np.stack([rgb_histogram_tf(p) for p in val_df.file_path])
train_y = train_df.label.values
val_y   = val_df.label.values

# Training and evaluation
knn = KNeighborsClassifier(n_neighbors=NEIGHBORS, metric='euclidean')
knn.fit(train_X, train_y)
val_pred = knn.predict(val_X)
acc = accuracy_score(val_y, val_pred)
cm  = confusion_matrix(val_y, val_pred)
print(f"[k‑NN] k={NEIGHBORS}  val_accuracy={acc:.4f}")
print("[k‑NN] confusion matrix:\n", cm)


# Saving the model
joblib.dump(knn, "knn_colorhist.pkl")
print("[k‑NN] wrote knn_colorhist.pkl")
