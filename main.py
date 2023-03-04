import cv2
import numpy as np
from keras.models import load_model

# ładuj model
model = load_model('emotions_detection_fer_model.h5')

# otwórz kamerę
cap = cv2.VideoCapture(0)

emotion_dict = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

# Wczytanie kaskady klasyfikatorów Haar do wykrywania twarzy
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def rectangleBackground():
    # Utworzenie przezroczystego prostokąta
    color = (0, 0, 0)  # kolor czarny
    x, y = 0, 0  # położenie lewego górnego rogu
    w, h = 210, 260  # szerokość i wysokość
    alpha = 0.5  # wartość przeźroczystości
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)  # ustawienie przeźroczystości
    # Dodanie efektu przezroczystości
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def textDsplaying(i, probability):
    emotion = emotion_dict[i]
    probability = round(probability * 100, 2)
    emotion_text = f" {emotion}: {probability}%  "

    cv2.putText(frame, emotion_text, (10, 60 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

def faceRectangle():
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Wykrycie twarzy na ramce
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Rysowanie prostokąta wokół wykrytej twarzy
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


while True:
    # pobierz obraz z kamery
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48))
    gray = gray.astype("float32") / 255.0
    gray = np.expand_dims(gray, axis=0)
    gray = np.expand_dims(gray, axis=-1)

    # predykcja
    emotion_probability = model.predict(gray)[0]
    print("emotion_probability: ", emotion_probability)
    emotion_index = np.argmax(emotion_probability)

    # mapowanie indexu emocji na etykietę
    labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    domainEmotion = labels[emotion_index]

    rectangleBackground()
    faceRectangle()

    # wyświetlenie wyniku na obrazie
    text = f"Emotion: {domainEmotion}, \n probability: {emotion_probability}"
    cv2.putText(frame, domainEmotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    emotion_text = ""
    for i, probability in enumerate(emotion_probability):
        textDsplaying(i, probability)

    cv2.imshow("Emotion Recognition", frame)

    # zakończ program po wciśnięciu klawisza 'esc'
    key = cv2.waitKey(50)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()