from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation, BatchNormalization

# definicja modelu
model = Sequential()

# warstwy konwolucyjne
model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))
# model.add(Conv2D(32, (3, 3)))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
# model.add(Conv2D(64, (3, 3)))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
# model.add(Conv2D(128, (3, 3)))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
# model.add(Conv2D(64, (3, 3)))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, (3, 3)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.1))
# #
# model.add(Conv2D(256, (3, 3)))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
# model.add(MaxPooling2D((2, 2)))
# model.add(Dropout(0.2))

#warstwa spłaszczająca
model.add(Flatten())

# warstwa gęsta
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation("relu"))
#przedziwdziała przetrenowaniu -  losowo wybiera, które neurony będą ignorowane w trakcie jednej iteracji treningu
model.add(Dropout(0.5))

# model.add(Dense(128))
# model.add(BatchNormalization())
# model.add(Activation("relu"))
# model.add(Dropout(0.5))

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# warstwa wyjściowa
model.add(Dense(len(EMOTIONS), activation='softmax'))

# kompilowanie modelu używając optymalizatora Adam, funkcji straty (f. kosztu) categorical_crossentropy (dla problemów klasyfikacji wieloklasowej) i metryki accuracy (dokładności).
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# utworzenie generatora danych treningowych
data_gen = ImageDataGenerator(rescale=1./255)

# wczytanie danych treningowych za pomocą generatora - wykorzystanie generatora do wczytania danych treningowych
train_data = data_gen.flow_from_directory("train_fer", target_size=(48, 48), color_mode='grayscale', batch_size=32, class_mode='categorical')
# trenowanie modelu na danych treningowych przez 150 epok
model.fit(train_data, epochs=200)

model.save('emotions_detection_fer_model_new6_dense128_wiecej_konw.h5')