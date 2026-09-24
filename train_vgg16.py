import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from preprocess import get_data_generators

# Load data generators (automatically reads all new folders in dataset/Training)
train_gen, val_gen, test_gen = get_data_generators()
num_classes = len(train_gen.class_indices)

print(f"Total classes detected: {num_classes}")
print(f"Class mapping: {train_gen.class_indices}")

# Build VGG16 Transfer Learning Model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head matching the new number of classes
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("--- Training Advanced Multi-Class VGG16 Model ---")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5
)

# Save the updated model
model.save('models/vgg16_model.h5')
print("Advanced Multi-Class VGG16 Model saved successfully as 'models/vgg16_model.h5'!")