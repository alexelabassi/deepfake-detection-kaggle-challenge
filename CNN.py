import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow_addons as tfa
from tensorflow.keras import regularizers


# Reading data
BASE_DIR       = r"data"
TRAIN_IMG_DIR  = os.path.join(BASE_DIR, "train")
VAL_IMG_DIR    = os.path.join(BASE_DIR, "validation")
TEST_IMG_DIR   = os.path.join(BASE_DIR, "test")

TRAIN_CSV      = os.path.join(BASE_DIR, "train.csv")
VAL_CSV        = os.path.join(BASE_DIR, "validation.csv")
TEST_CSV       = os.path.join(BASE_DIR, "test.csv")
SUB_CSV        = os.path.join(BASE_DIR, "sample_submission.csv")

# Hyperparameters
IMG_HEIGHT    = 160
IMG_WIDTH     = 160
BATCH_SIZE    = 64
NUM_CLASSES   = 5
EPOCHS        = 100
LEARNING_RATE = 1e-3

# Reading CSVs
train_df = pd.read_csv(TRAIN_CSV)    # [image_id, label]
val_df   = pd.read_csv(VAL_CSV)      # [image_id, label]
test_df  = pd.read_csv(TEST_CSV)     # [image_id]

# File paths
train_df["file_path"] = train_df["image_id"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x + ".png"))
val_df["file_path"]   = val_df["image_id"].apply(lambda x: os.path.join(VAL_IMG_DIR, x + ".png"))
test_df["file_path"]  = test_df["image_id"].apply(lambda x: os.path.join(TEST_IMG_DIR, x + ".png"))

# Processing & Augmentation
def parse_image(filename, label=None, is_training=False):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_png(image_string, channels=3)
    # Resize
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    # Normalization to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0

    if is_training:
        # Flip
        image = tf.image.random_flip_left_right(image)
        # Brightness
        image = tf.image.random_brightness(image, max_delta=0.1)
        # Contrast
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        # Rotation
        angle = tf.random.uniform([], -15.0, 15.0) * tf.constant(3.14159265/180.0)
        image = tfa.image.rotate(image, angles=angle, fill_mode='reflect')
        # Zoom
        scales = tf.random.uniform([2], 0.9, 1.1)
        h_scaled = tf.cast(IMG_HEIGHT * scales[0], tf.int32)
        w_scaled = tf.cast(IMG_WIDTH  * scales[1], tf.int32)
        image = tf.image.resize(image, [h_scaled, w_scaled])
        image = tf.image.resize_with_crop_or_pad(image, IMG_HEIGHT, IMG_WIDTH)
        # Shift
        tx = tf.random.uniform([], -0.1 * IMG_WIDTH, 0.1 * IMG_WIDTH)
        ty = tf.random.uniform([], -0.1 * IMG_HEIGHT, 0.1 * IMG_HEIGHT)
        image = tfa.image.translate(image, [tx, ty], fill_mode='reflect')

    if label is None:
        return image
    return image, label



# Training Data
train_paths  = train_df["file_path"].values
train_labels = train_df["label"].values

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
train_ds = train_ds.shuffle(buffer_size=len(train_paths))
train_ds = train_ds.map(
    lambda x, y: parse_image(x, y, is_training=True),
    num_parallel_calls=tf.data.AUTOTUNE
)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Validation Data
val_paths  = val_df["file_path"].values
val_labels = val_df["label"].values

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
val_ds = val_ds.map(
    lambda x, y: parse_image(x, y, is_training=False),
    num_parallel_calls=tf.data.AUTOTUNE
)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Test Data
test_paths = test_df["file_path"].values

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(
    lambda x: parse_image(x, label=None, is_training=False),
    num_parallel_calls=tf.data.AUTOTUNE
)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# CNN Model
def build_deepfake_cnn(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES):
    model = models.Sequential()

# Block 1: 32 filters
    model.add(layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    # Block 2: 64 filters
    model.add(layers.Conv2D(64, (3,3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    # Block 3: 128 filters
    model.add(layers.Conv2D(128, (3,3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    # Block 4: 256 filters
    model.add(layers.Conv2D(256, (3,3), activation="relu", padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    # Global Average Pooling
    model.add(layers.GlobalAveragePooling2D())

    # Dense block 1
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.6))

    # Dense block 2
    model.add(layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))

    # Last layer for classification
    model.add(layers.Dense(num_classes, activation="softmax", kernel_regularizer=regularizers.l2(1e-4)))

    return model

model = build_deepfake_cnn()
model.summary()

# Compilation
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks, early stopping and saving the best model
checkpoint_cb = ModelCheckpoint(
    "best_deepfake_cnn.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

earlystop_cb = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_cb = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    verbose=1
)

# Training
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[checkpoint_cb, earlystop_cb, reduce_lr_cb]
)

# Final evaluation on validation set
val_loss, val_acc = model.evaluate(val_ds)
print(f"Final validation accuracy: {val_acc:.4f}")

# Prediction on test set in submission.csv
model.load_weights("best_deepfake_cnn.h5")
pred_probs = model.predict(test_ds, verbose=1)
pred_labels = pred_probs.argmax(axis=1)

submission = pd.read_csv(SUB_CSV)   
submission["label"] = pred_labels
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
