import os
import cv2
import numpy as np
from keras.models import load_model

# Wczytanie wytrenowanej sieci neuronowej
model = load_model('emotions_detection_fer_model_new2.h5')

# Utworzenie listy emocji
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Utworzenie słownika, aby przyporządkować każdej emocji indeks
emotion_dict = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sad': 5, 'surprise': 6}

# Ścieżka do folderu z testowymi zdjęciami
test_dir = 'test_fer'

# Utworzenie słownika, aby przechowywać liczniki poprawnych odpowiedzi dla każdej emocji
correct_predictions_dict = {emotion: 0 for emotion in emotions}

# Utworzenie słownika, aby przechowywać liczbę zdjęć dla każdej emocji
image_counts_dict = {emotion: 0 for emotion in emotions}

# Przetestowanie sieci neuronowej na każdym zdjęciu w folderze testowym
for root, dirs, files in os.walk(test_dir):
    for file in files:
        # Wczytanie zdjęcia
        img = cv2.imread(os.path.join(root, file))

        # Przekształcenie zdjęcia na czarno-białe
        img = cv2.imread(os.path.join(root, file), cv2.IMREAD_GRAYSCALE)

        # Przekształcenie zdjęcia do wymaganego rozmiaru (np. 48x48 pikseli)
        img = cv2.resize(img, (48, 48))

        # Normalizacja pikseli do zakresu od 0 do 1
        img = img / 255.0

        # Przekształcenie zdjęcia na wektor 1D
        img = np.expand_dims(img, axis=0)

        # Przewidywanie emocji za pomocą sieci neuronowej
        pred = model.predict(img)

        # Wybór emocji z najwyższą wartością prawdopodobieństwa
        max_index = np.argmax(pred[0])

        # Przypisanie emocji do indeksu
        predicted_emotion = emotions[max_index]

        # Wyciągnięcie prawdziwej emocji z nazwy folderu
        true_emotion = root.split(os.path.sep)[-1]

        # Sprawdzenie, czy sieć poprawnie rozpoznała emocję
        if predicted_emotion == true_emotion:
            correct_predictions_dict[true_emotion] += 1

        image_counts_dict[true_emotion] += 1

# Obliczenie dokładności sieci neuronowej dla każdej emocji oddzielnie
for emotion in emotions:
    accuracy = correct_predictions_dict[emotion] / image_counts_dict[emotion]
    print(f"Dokładność sieci neuronowej dla emocji {emotion}: {accuracy * 100:.2f}%")

# Obliczenie ogólnej dokładności sieci neuronowej
overall_accuracy = sum(correct_predictions_dict.values()) / sum(image_counts_dict.values())
print(f"Ogólna dokładność sieci neuronwej:  {overall_accuracy * 100:.2f}%")
