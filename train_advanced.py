import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils import class_weight

# --- QUALITY CONFIGURATION ---
DATA_DIR = r"c:/Users/garim/OneDrive/Desktop/ML_plant/PlantVillage"
MODEL_SAVE_PATH = "plant_disease_model.h5"
CLASSES_SAVE_PATH = "class_names.json"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 15 # Extended for precision

print("--- STARTING HIGH-PRECISION RELIABILITY RECOVERY ---")

# 1. Advanced Data Loading
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

class_names = train_ds.class_names
with open(CLASSES_SAVE_PATH, 'w') as f:
    json.dump(class_names, f)

# 2. CALIBRATING CLASS WEIGHTS (Bias Mitigation)
print(">>> Calibrating class weights to eliminate Tomato Bias...")
y_train = np.concatenate([y for x, y in train_ds], axis=0)
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(weights))
print(f">>> Weights applied: {len(class_weight_dict)} classes calibrated.")

# 3. HIGH-PRECISION PIPELINE
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

# Build complete model architecture
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x) # Higher capacity head
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

# 4. PHASE 1: Classification Head Solidification
print("PHASE 1: Calibrating Classification Head (Frozen Core)...")
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
    ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max')
]

model.fit(
    train_ds,
    epochs=PHASE1_EPOCHS,
    validation_data=val_ds,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# 5. PHASE 2: Deep Morphological Fine-Tuning
print("PHASE 2: Deep Fine-Tuning for Maximum Precision...")
base_model.trainable = True
# Unfreeze top 50 layers for deeper feature extraction
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-5), # Ultra-low LR for stability
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.fit(
    train_ds,
    epochs=PHASE2_EPOCHS,
    validation_data=val_ds,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

print("\n🚀 RELIABILITY RECOVERY COMPLETE.")
print(f"Final Model Saved: {MODEL_SAVE_PATH}")
