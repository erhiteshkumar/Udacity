import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


lines = []
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
count = 0
images = []
measurements = []
del lines[0]
for line in lines:
	for i in range(3):
		source_path =  line[i]
		filename = source_path.split('/')[-1]
		current_path = './data/IMG/' + filename
		image = cv2.imread(current_path)
		#print(current_path)
		images.append(image)
		try:
			measurement = float(line[3])
			measurements.append(measurement)
		except ValueError:
			print("error on line",line[3])

augmented_images, augmented_measurments = [] , []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurments.append(measurement)
	augmented_images.append(cv2.flip(image, 1))
	augmented_measurments.append(measurement *-1.0)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurments)


model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2,shuffle=True, nb_epoch=5)
model.save('model.h5')