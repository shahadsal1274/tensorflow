import os
import sys
import subprocess

required_libraries = ['tensorflow', 'numpy', 'librosa', 'scikit-learn']
for library in required_libraries:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', library])

import tensorflow as tf
import numpy as np
import librosa
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

audio_paths = []
sentiments = []
with open('audio_data.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        audio_paths.append(row[0])
        sentiments.append(row[1])

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        return mfccs.flatten()
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros(13 * 43)

features = np.array([extract_features(path) for path in audio_paths], dtype=object)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(sentiments)

max_length = max(len(f) for f in features)
features = np.array([np.pad(f, (0, max_length - len(f)), 'constant') for f in features])

X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(max_length,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

results = model.predict(X_test)

np.savetxt('نتائج.txt', results, delimiter=',')

try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Model saved as 'model.tflite'")
except Exception as e:
    print(f"Error converting model to TFLite: {e}")

# حفظ النموذج بتنسيق TFLite
#converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()

# with open('model.tflite', 'wb') as f:
#     f.write(tflite_model)

# print("Model saved as 'model.tflite'")