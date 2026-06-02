import tensorflow as tf
from tensorflow.keras import layers, models
import os

def create_bootstrap_model(output_path):
    print("Building model architecture based on plant_disease_model.ipynb...")
    
    # 1. Base Model (MobileNetV2 as used in notebook cell 15)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), 
        include_top=False, 
        weights='imagenet'
    )
    base_model.trainable = False  # Keep frozen as a bootstrap
    
    # 2. Custom Head (based on notebook cell 17)
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256),
        layers.BatchNormalization(), # Added for stability
        layers.Activation('relu'),
        layers.Dropout(0.2),
        layers.Dense(15, activation='softmax')
    ])
    
    print("Compiling model...")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Saving bootstrap model to: {output_path}")
    # Save as .h5 as requested by the backend configuration
    model.save(output_path)
    print(">>> SUCCESS: Bootstrap model generated.")

if __name__ == "__main__":
    target = os.path.join(os.path.dirname(__file__), "plant_disease_model.h5")
    create_bootstrap_model(target)
