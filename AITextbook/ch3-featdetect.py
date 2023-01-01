#Convolutional neural network for feature detection instead of just focusing on the pixels
#Convolution is a filter of weights that are used to multply a pixel with its neighbors to get a new value
#Different filters emphasize different aspects of images - can learn the best filters to match inputs to outputs
#Combine ths with pooling to reduce amount of info in an image while maintaining the important features - 
# Pooling is reducing pixels while maintaining semantics of the content - groups of pixels get replaced with one 
import tensorflow as tf
data = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = data.load_data()

#Have to reshape the images to be 28x28x1 because Conv2D is designe for color images - adding the one shows its grayscale
training_images = training_images.reshape(60000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images / 255.0

#First we do the convolutions and pooling (2 rounds of it) - don't need to flatten until after 
#Parameters in Conv2D are the number of convs to learn (64) and the size of the convs (3x3)
#Parameters in MaxPool are the size of the pools (2x2)
#Then pass to the Dense layer and it continues as in the last chapter
model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu', 
                  input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
      tf.keras.layers.MaxPooling2D(2,2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation=tf.nn.relu),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=25)

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])
#Shows shape/number of parameters at each step
model.summary()

#%%
import tensorflow as tf
#Retrieving/unzipping the horse or human images (training data set) already presorted into horse/human folders
import urllib.request
import zipfile
from tensorflow.keras.optimizers import RMSprop

file_name = "horse-or-human.zip"
training_dir = 'horse-or-human/training/'

#uncomment if need to re-unzip
#zip_ref = zipfile.ZipFile(file_name, 'r')
#zip_ref.extractall(training_dir)
#zip_ref.close()

#This package takes the images from the unzipped directories - 300x300 is img size and binary is b/c there are two categories
#It assigns labels based on the folders they were in (this dataset doesn't come with labels)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
  training_dir,
  target_size=(300, 300),
  class_mode='binary'
)

#Doing the same for the validation dataset - diff from test data set because it shows you how
# the model does with unseen data at each epoch instead of after training is done
validation_file_name ="validation-horse-or-human.zip"
validation_dir = 'horse-or-human/validation/'

#uncomment if need to re-unzip
#zip_ref = zipfile.ZipFile(validation_file_name, 'r')
#zip_ref.extractall(validation_dir)rm
#zip_ref.close()

validation_datagen = ImageDataGenerator(rescale=1/255)

validation_generator = validation_datagen.flow_from_directory(
  validation_dir,
  target_size=(300, 300),
  class_mode='binary'
)

#Need more layers for this network b/c imgs are larger
#Also imgs are color so have 3 channels instead of 1
#Only need one output neuron b/c it is a binary classification - sigmoid function will make values go towards either 1 or 0
#This number of layers will make the items passed to the Dense layer much smaller and simple, but # of parameters much 
# higher so its much slower
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu' , input_shape=(300, 300, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])

#Can take out the validation data parameter if just want to traiin like normal
history = model.fit(train_generator, epochs=5, validation_data=validation_generator)

#TESTING - would involve using the model on real images of horses or humans (not CGI like in training) and seeing how it performs
#When done in the text, we see that it works for some images but not othes - this is because there are just some poses or angles 
# that the model wasn't trained on and therefore can't recognize
#One way to help get around this without using an insanely large dataset is through image augmentation - involves changing the images
# in some way before running the model (rotating, shifting, shearing, zooming, etc.) in order to increase what the model is exposed to
#Would do it when importing images with ImageDataGenerator - instead of just rescaling by dividing by 255, also apply these other transforms
#The text shows that doing this allows the model to get real images correct that it couldn't before even though accuracy during training 
# went down (shows that before it was overfitting to training data)

#%%
#Another concept that models can take advantage of is transfer learning - using filters that have been learned by models already
# trained on enormous datasets - allows models to benefit from all that training without actually doing it
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop

weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_file = "inception_v3.h5"
urllib.request.urlretrieve(weights_url, weights_file)

pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                include_top=False,
                weights=None)

pre_trained_model.load_weights(weights_file)

pre_trained_model.summary()
# %%
#Making sure to freeze the network so it doesn't retrain and then choosing one of the layers as the one to feed into my network
for layer in pre_trained_model.layers:
    layer.trainable = False

# pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
#Dropout is done to help combat overfitting - can happen when neurons end up with similar weights and biases and eventually
#the whole model becomes too specialized to certain inputs
#Dropout randomly removes a certain percentage of neurons to prevent their influence from causing this
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)

model2 = Model(pre_trained_model.input, x)

model2.compile(optimizer=RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['acc'])


train_generator2 = train_datagen.flow_from_directory(
  training_dir,
  batch_size=20,
  target_size=(150, 150),
  class_mode='binary'
)

validation_generator2 = validation_datagen.flow_from_directory(
  validation_dir,
  batch_size=20,
  target_size=(150, 150),
  class_mode='binary'
)
history2 = model2.fit(train_generator2, epochs=10, validation_data=validation_generator2)

# %%
#Multiclass classification is pretty much the same thing with a few changes - need more output neurons, different loss function
# but everything else almost identical

