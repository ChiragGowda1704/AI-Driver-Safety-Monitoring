import os
import cv2
import numpy as np
import argparse
import warnings
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="train", help="Mode: train or display")
args = parser.parse_args()
mode = args.mode.lower()

def plot_model_history(model_history):
    """Plot Accuracy and Loss curves given the model_history"""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axs[0].plot(model_history.history['accuracy'], label='Train')
    axs[0].plot(model_history.history['val_accuracy'], label='Validation')
    axs[0].set_title('Model Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    # Loss
    axs[1].plot(model_history.history['loss'], label='Train')
    axs[1].plot(model_history.history['val_loss'], label='Validation')
    axs[1].set_title('Model Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig('plot.png')
    plt.show()

# Build CNN model
def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    return model

model = build_model()

if mode == "train":
    train_dir = 'data/train'
    val_dir = 'data/test'
    num_train, num_val = 28709, 7178
    batch_size, epochs = 64, 50

    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(48, 48), batch_size=batch_size,
        color_mode="grayscale", class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=(48, 48), batch_size=batch_size,
        color_mode="grayscale", class_mode='categorical'
    )

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.0001, decay=1e-6),
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=num_val // batch_size
    )

    plot_model_history(history)
    model.save_weights('model.h5')

elif mode == "display":
    model.load_weights('model.h5')
    cv2.ocl.setUseOpenCL(False)

    emotion_dict = {
        0: "Angry", 1: "Disgusted", 2: "Fearful",
        3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
    }

    face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_resized = cv2.resize(roi_gray, (48, 48))
            roi_expanded = np.expand_dims(np.expand_dims(roi_resized, -1), 0)

            prediction = model.predict(roi_expanded, verbose=0)
            max_index = int(np.argmax(prediction))

            label = emotion_dict[max_index]
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            cv2.putText(frame, label, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame, (960, 640)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print("Invalid mode. Use '--mode train' or '--mode display'.")
