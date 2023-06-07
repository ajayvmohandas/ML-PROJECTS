import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("fer2013.csv")

df.shape

df.head(10)

df.emotion.unique()

dff=df.drop_duplicates(subset="emotion")
dff.head(10)

for index, row in dff.iterrows():
    emotion=row["emotion"]
    emotion_pixels=row["pixels"]
    
    #Convert the pixel value into a numpy array
    pixel_array=np.array(list(map(int,emotion_pixels.split())))
    
    image_shape=(48,48)
    image=pixel_array.reshape(image_shape)
    
    plt.imshow(image,cmap="gray")
    plt.title(f"Emotion: {emotion}")
    plt.axis("off")
    plt.show()
    
emotion_label={0:"anger", 1:"disgust", 2:"fear", 3:"happiness", 4:"sadness", 5:"surprise", 6: "neutral"}

df.emotion.value_counts()

sns.countplot(df.emotion)

import math

math.sqrt(len(df.pixels[0].split(" ")))

fig=plt.figure(figsize=(14,14))
k=0
for i in sorted(df.emotion.unique()):
    for j in range(7):
        px=df[df.emotion==i].pixels.iloc[k]
        px=np.array(px.split(" ")).reshape(48,48).astype("float32")
        k+=1
        plot=plt.subplot(7,7,k)
        plot.imshow(px,cmap="gray")
        plot.set_xticks([])
        plot.set_yticks([])
        plot.set_title(emotion_label[i])
        plt.tight_layout()

#New labels
new_labels=[3,4,6]
df=df[df.emotion.isin(new_labels)]

df.shape

img_array=df.pixels.apply(lambda x:np.array(x.split(" ")).reshape(48,48,1).astype("float32"))
img_array=np.stack(img_array, axis=0)

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils import np_utils
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU, Activation
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from concurrent.futures import ThreadPoolExecutor

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus, 'GPU')

le=LabelEncoder()
img_labels=le.fit_transform(df.emotion)
img_labels=np_utils.to_categorical(img_labels)
img_labels.shape

le_name_mapping=dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

X_train, X_test, y_train,y_test = train_test_split(img_array,img_labels,shuffle=True,stratify=img_labels,test_size=0.1, random_state=42)
X_train.shape,X_test.shape,y_train.shape,y_test.shape

img_width=X_train.shape[1]
img_height=X_train.shape[2]
img_depth=X_train.shape[3]
num_classes=y_train.shape[1]

X_train=X_train/255
X_test=X_test/255

def build_model(optim):
    model=Sequential(name="DCNN")
    
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(5,5),
            input_shape=(img_width,img_height, img_depth),
            activation="elu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_1"
        )
    )
    model.add(BatchNormalization(name="batchnorm_1"))
    
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(5,5),
            activation="elu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_2"
        )
    )
    model.add(BatchNormalization(name="batchnorm_2"))
    
    model.add(MaxPooling2D(pool_size=(2,2), name="maxpool2d_1"))
    model.add(Dropout(0.4, name="dropout_1"))
    
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation="elu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_3"
        )
    )
    model.add(BatchNormalization(name="batchnorm_3"))
        
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            activation="elu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_4"
        )
    )
    model.add(BatchNormalization(name="batchnorm_4"))
    
    model.add(MaxPooling2D(pool_size=(2,2), name="maxpool2d_2"))
    model.add(Dropout(0.4, name="dropout_2"))
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3,3),
            activation="elu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_5"
        )
    )        
    model.add(BatchNormalization(name="batchnorm_5"))
    
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(3,3),
            activation="elu",
            padding="same",
            kernel_initializer="he_normal",
            name="conv2d_6"
        )
    ) 
    model.add(BatchNormalization(name="batchnorm_6"))
    
    model.add(MaxPooling2D(pool_size=(2,2), name="maxpool2d_3"))
    model.add(Dropout(0.5, name="dropout_3"))
    model.add(Flatten(name="Flatten"))
    model.add(
        Dense(
            units=128,
            activation="elu",
            kernel_initializer="he_normal",
            name="dense_1"
        )
    )        
    model.add(BatchNormalization(name="batchnorm_7"))
    
    model.add(Dropout(0.5, name="dropout_4"))
    
    model.add(
        Dense(
            num_classes,
            activation="softmax",
            name="out_layer"
        )
    ) 
    
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optim,
        metrics=["accuracy"]
    )
    
    model.summary()
    
    return model

early_stopping=EarlyStopping(
    monitor="val_accuracy",
    min_delta=0.00005,
    patience=11,
    verbose=1,
    restore_best_weights=True,
)

lr_scheduler=ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.5,
    patience=7,
    min_lr=1e-7,
    verbose=1,
)

callbacks=[
    early_stopping,
    lr_scheduler
]

train_datagen=ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True
)
train_datagen.fit(X_train)

import psutil

# Get the number of logical CPUs
num_cpus = psutil.cpu_count(logical=True)

# Check if specific number of CPUs are available
desired_cpus = 4  # Update with the desired number of CPUs
cpus_available = num_cpus >= desired_cpus

# Print the availability of the CPUs
print(f"Desired CPUs ({desired_cpus}) available: {cpus_available}")

import multiprocessing
from keras import backend as K

batch_size = 32
epochs = 100

def train_batch(batch):
    x_batch, y_batch = batch
    with K.tf.device('/cpu:0'):  # Use CPU for training to avoid GPU memory conflicts
        return model.train_on_batch(x_batch, y_batch)

def train_model(model, train_data, val_data, batch_size, epochs, callbacks):
    pool = multiprocessing.Pool()

    history = model.fit(
        x=train_data[0],
        y=train_data[1],
        validation_data=val_data,
        batch_size=batch_size,
        steps_per_epoch=len(train_data[0]) // batch_size,
        epochs=epochs,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=pool._processes,  # Use the number of available processes
        verbose=1
    )

    pool.close()
    pool.join()

    return history

optims = [
    optimizers.Nadam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam"),
    optimizers.Adam(0.001)
]

model = build_model(optims[0])

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True
)
train_datagen.fit(X_train)

train_data = (X_train, y_train)
val_data = (X_test, y_test)

history = train_model(model, train_data, val_data, batch_size, epochs, callbacks)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

model.save("model.h5")

# Plot training loss and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training accuracy and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
