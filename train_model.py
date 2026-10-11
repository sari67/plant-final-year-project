import tensorflow as tf
import tensorflow_hub as hub
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ==================================================
# DATASET PATH
# ==================================================
BASE_DIR = r"D:\project\dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "val")

# ==================================================
# MODEL SAVE PATH
# ==================================================
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_model_4.h5")
CLASS_PATH = os.path.join(MODEL_DIR, "class_names_4.txt")
os.makedirs(MODEL_DIR, exist_ok=True)

# ==================================================
# IMAGE PARAMETERS
# ==================================================
IMG_SIZE = (224,224)
BATCH_SIZE = 16
EPOCHS = 20

# ==================================================
# DATA AUGMENTATION
# ==================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8,1.2],
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# ==================================================
# CLASS NAMES
# ==================================================
class_names = list(train_data.class_indices.keys())
print("\nTotal Classes:", len(class_names))
print("Classes:", class_names)

# ==================================================
# INPUT LAYER
# ==================================================
inputs = Input(shape=(224,224,3))

# ==================================================
# EfficientNet-Lite0 Feature Extractor (Frozen)
# ==================================================
feature_extractor = hub.KerasLayer(
    "model/efficientnet_lite0",  # TF1 module
    trainable=False              # Must stay False
)
x = feature_extractor(inputs)

# ==================================================
# CLASSIFICATION HEAD
# ==================================================
x = Dense(256, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)  # reduced from 0.5 to reduce underfitting

x = Dense(128, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)  # reduced from 0.3

outputs = Dense(train_data.num_classes, activation="softmax")(x)
model = Model(inputs, outputs)

# ==================================================
# COMPILE MODEL
# ==================================================
loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
model.compile(
    optimizer=Adam(learning_rate=3e-4),  # slightly higher learning rate
    loss=loss,
    metrics=["accuracy"]
)

model.summary()

# ==================================================
# CALLBACKS
# ==================================================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-6
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

callbacks = [early_stop, reduce_lr, checkpoint]

# ==================================================
# TRAIN MODEL (Stage 1 only, Frozen Feature Extractor)
# ==================================================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ==================================================
# SAVE CLASS NAMES
# ==================================================
with open(CLASS_PATH,"w") as f:
    for name in class_names:
        f.write(name+"\n")

print("\nTraining Completed Successfully")
print("Model saved at:", MODEL_PATH)
print("Class names saved at:", CLASS_PATH)
