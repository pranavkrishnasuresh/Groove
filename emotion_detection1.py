import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Flatten, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator , img_to_array, load_img
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.losses import categorical_crossentropy

# Working with pre trained model

base_model = MobileNet( input_shape=(224,224,3), include_top= False )

for layer in base_model.layers:
  layer.trainable = False


x = Flatten()(base_model.output)
x = Dense(units=7 , activation='softmax' )(x)

# creating our model
model = Model(base_model.input, x)

model.compile(optimizer='adam', loss= categorical_crossentropy , metrics=['accuracy']  )

train_datagen = ImageDataGenerator(
     zoom_range = 0.2,
     shear_range = 0.2,
     horizontal_flip=True,
     rescale = 1./255
)

train_data = train_datagen.flow_from_directory(directory= "image-classifier-main\\train",
                                               target_size=(224,224),
                                               batch_size=32,
                                  )


train_data.class_indices

val_datagen = ImageDataGenerator(rescale = 1./255 )

val_data = val_datagen.flow_from_directory(directory= "image-classifier-main\\test",
                                           target_size=(224,224),
                                           batch_size=32,
                                  )

t_img , label = train_data.next()

def plotImages(img_arr, label):
  """
  input  :- images array
  output :- plots the images
  """
  count = 0
  for im, l in zip(img_arr,label) :
    plt.imshow(im)
    plt.title(im.shape)
    plt.axis = False
    plt.show()

    count += 1
    if count == 10:
      break

plotImages(t_img, label)

from keras.callbacks import ModelCheckpoint, EarlyStopping

es = EarlyStopping(monitor='val_accuracy', min_delta= 0.01 , patience= 5, verbose= 1, mode='auto')

mc = ModelCheckpoint(filepath="best_model.h5", monitor= 'val_accuracy', verbose= 1, save_best_only= True, mode = 'auto')

call_back = [es, mc]

hist = model.fit_generator(train_data,
                           steps_per_epoch= 5,
                           epochs= 5,
                           validation_data= val_data,
                           validation_steps= 8,
                           callbacks=[es,mc])

from keras.models import load_model
model = load_model("best_model.h5")

h =  hist.history
h.keys()

plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'] , c = "red")
plt.title("acc vs v-acc")
plt.show()

plt.plot(h['loss'])
plt.plot(h['val_loss'] , c = "red")
plt.title("loss vs v-loss")
plt.show()

op = dict(zip( train_data.class_indices.values(), train_data.class_indices.keys()))

def predict_emotion(file_path):
    img = load_img(file_path, target_size=(224,224))
    i = img_to_array(img) / 255
    input_arr = np.array([i])

    pred = np.argmax(model.predict(input_arr))

    op = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    return op[pred]