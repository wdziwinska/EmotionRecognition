from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# definicja modelu
model = Sequential()

# warstwa konwolucyjna
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D((2, 2)))

# warstwa konwolucyjna
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# warstwa konwolucyjna
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# warstwa konwolucyjna
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# warstwa wypłaszczająca
model.add(Flatten())

# warstwa ukryta
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# warstwa wyjściowa
model.add(Dense(len(EMOTIONS), activation='softmax'))

# kompilowanie modelu używając optymalizatora Adam, funkcji straty categorical_crossentropy (dla problemów klasyfikacji wieloklasowej) i metryki accuracy (dokładności).
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# utworzenie generatora danych treningowych
data_gen = ImageDataGenerator(rescale=1./255)

# wczytanie danych treningowych za pomocą generatora - wykorzystanie generatora do wczytania danych treningowych
train_data = data_gen.flow_from_directory("train_fer", target_size=(48, 48), color_mode='grayscale', batch_size=32, class_mode='categorical')
# trenowanie modelu na danych treningowych przez 50 epok
model.fit(train_data, epochs=150)

model.save('emotions_detection_fer_model.h5')