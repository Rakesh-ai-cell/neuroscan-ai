import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocess import get_data_generators
import os

_, _, test_gen = get_data_generators()

# List of all algorithms and their saved file paths
models_to_evaluate = [
    ('VGG16', 'models/vgg16_model.h5'),
    ('ResNet50', 'models/resnet50_model.h5'),
    ('MobileNetV2', 'models/mobilenet_model.h5'),
    ('DenseNet121', 'models/densenet_model.h5'),
    ('EfficientNetB0', 'models/efficientnet_model.h5')
]

print("==========================================")
print("     ALGORITHM PERFORMANCE COMPARISON     ")
print("==========================================")

for name, path in models_to_evaluate:
    if os.path.exists(path):
        model = load_model(path)
        loss, accuracy = model.evaluate(test_gen, verbose=0)
        print(f"Algorithm: {name:15} | Test Accuracy: {accuracy * 100:6.2f}% | Loss: {loss:.4f}")
    else:
        print(f"Algorithm: {name:15} | Status: Model file not found at {path}")

print("==========================================")