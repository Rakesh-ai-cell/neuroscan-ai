import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2, DenseNet121, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from preprocess import get_data_generators
import os

# 1. Load data generators (assuming 48 classes)
train_gen, val_gen, test_gen = get_data_generators()
num_classes = len(train_gen.class_indices)

print(f"Detected {num_classes} classes for training.")

# 2. Define the different architectures to train
base_models = {
    'vgg16': VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3)),
    'resnet50': ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3)),
    'mobilenet': MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3)),
    'densenet': DenseNet121(weights='imagenet', include_top=False, input_shape=(150, 150, 3)),
    'efficientnet': EfficientNetB0(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
}

os.makedirs('models', exist_ok=True)

# 3. Training loop for each model
for model_name, base_model in base_models.items():
    print(f"\n==========================================")
    print(f"   TRAINING {model_name.upper()} MODEL")
    print(f"==========================================")
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
        
    # Add custom classification head for 48 classes
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Configure Early Stopping (stops if validation loss doesn't improve for 5 epochs)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # Train the model up to 25 epochs with early stopping
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=25,
        callbacks=[early_stopping]
    )
    
    # Save the trained model
    save_path = f"models/{model_name}_model.h5"
    model.save(save_path)
    print(f"--- {model_name.upper()} model successfully saved to {save_path} ---")

print("\nAll models have finished training with Early Stopping!")