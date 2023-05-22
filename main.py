import cv2
import numpy as np
from keras.models import load_model

def rectangle_background(x, y, w, h):
    # Utworzenie przezroczystego prostokąta
    color = (0, 0, 0)  # kolor czarny
    alpha = 0.5  # wartość przeźroczystości
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)  # ustawienie przeźroczystości
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # dodanie efektu przezroczystości

def top_3_emotions():
    emotion_probabilities = []
    for i, probability in enumerate(prediction):
        emotion = emotion_dictionary[i]
        probability = round(probability * 100, 2)
        emotion_probabilities.append((emotion, probability))

    sorted_emotion_probabilities = sorted(emotion_probabilities, key=lambda x: x[1], reverse=True)

    # Wybierz trzy emocje z najwyższymi wartościami predicted probabilities
    top_emotions = sorted_emotion_probabilities[:3]

    for i, (emotion, probability) in enumerate(top_emotions):
        emotion_text = f"{emotion}: {probability}%  "
        font_size = w / 285
        cv2.putText(frame, emotion_text, (x, y + h + 20 + int(h/8) + i * int(h/6)), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1,
                    cv2.LINE_AA)

model = load_model('emotions_detection_model_fer.h5')
cap = cv2.VideoCapture(0)
emotion_dictionary = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
face_cascade= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    # pobieranie obrazu z kamery
    ret, frame = cap.read()

    emotion_text = ""

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Wykrycie twarzy na ramce. detectMultiScale zwraca x, y, w, h
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Wykrycie każdej twarzy
    for (x, y, w, h) in faces:
        # wycięcie fragmentu obrazu, samą twarz
        face = gray[y:y + h, x:x + w]
        # przekształcenie obrazu
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        # Przetwarzany fragment obrazu zostaje przekształcaon do postaci tenosora. 1 - przetwarzana jest tylko jedna twarz na raz, 48 i 48 to szewrokość i wysokość, 1 - obraz jest w skali szarości
        face = face.reshape(1, 48, 48, 1)
        # Uzyskanie predykcji dla danej twarzy. [0] oznacza, że pobierana jest pierwsza wartość tensora.
        prediction = model.predict(face)[0]
        max_index = int(np.argmax(prediction))

        font_size = w / 225
        rectangle_background(x, y - 10, w, - int(h / 5))
        cv2.putText(frame, emotion_dictionary[max_index],
                    (x + 10, y - 20), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                    (255, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (255, 255, 255), 2)

        rectangle_background(x, y + h + 20, w, int(h / 2))
        top_3_emotions()

    cv2.imshow("Emotion Recognition", frame)

    # zakończenie programu po wciśnięciu klawisza 'esc'
    key = cv2.waitKey(50)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
