"""Trains a convolutional neural network to classify facial expressions from the FER2013 dataset.

The model architecture consists of multiple convolutional and max pooling layers to extract features, 
followed by fully connected layers and a softmax output layer to classify into 7 emotions.

The model is trained using the Adam optimizer and categorical cross entropy loss. Training data is 
preprocessed and augmented using Keras ImageDataGenerator. 

The trained model structure and weights are saved to disk for later use.
"""

# import required packages
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
# Conv2D -- extract ftaure from the image 
# MaxPooling2D -- reduce the size of the feature map
        # Dense -- fully connected layer
        # Dropout -- prevent overfitting
        # Flatten -- convert the feature map to a 1D vector
        # Adam -- optimizer
        # categorical_crossentropy -- loss function
        # accuracy -- evaluation metric
        # categorical_crossentropy -- loss function
        # categorical_accuracy -- evaluation metric
        # 256 -- number of filters
        # 3 -- number of convolutional layers
        # 128 -- number of neurons in the first convolutional layer
        # 64 -- number of neurons in the second convolutional layer
        # 32 -- number of neurons in the third convolutional layer
        # 16 -- number of neurons in the fourth convolutional layer
        # 128 -- number of neurons in the fifth convolutional layer
        # 64 -- number of neurons in the sixth convolutional layer
        # 32 -- number of neurons in the seventh convolutional layer
        # 16 -- number of neurons in the eighth convolutional layer
        # 128 -- number of neurons in the ninth convolutional layer
        # 64 -- number of neurons in the tenth convolutional layer
        # 32 -- number of neurons in the eleventh convolutional layer
        # 16 -- number of neurons in the twelfth convolutional layer
        # 8 -- number of neurons in the thirteenth convolutional layer
        # 4 -- number of neurons in the fourteenth convolutional layer
        # 2 -- number of neurons in the fifteenth convolutional layer
        # 1 -- number of neurons in the sixteenth convolutional layer
        
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all test images
train_generator = train_data_gen.flow_from_directory(
        'data/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# Preprocess all train images
validation_generator = validation_data_gen.flow_from_directory(
        'data/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# create model structure
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

# emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6), metrics=['accuracy'])
emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])


# Train the neural network/model
emotion_model_info = emotion_model.fit_generator(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=70,
        validation_data=validation_generator,
        validation_steps=7178 // 64)

# save model structure in jason file
model_json = emotion_model.to_json()
with open("emotion_model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
emotion_model.save_weights('emotion_model.h5')

