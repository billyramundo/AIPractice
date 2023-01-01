#FASHION MNIST - training to give clothing items 1 of 10 labels
import tensorflow as tf
import numpy as np

data = tf.keras.datasets.fashion_mnist

(train_imgs, train_labels), (test_imgs, test_labels) = data.load_data()

#Normalizing the images - improves performance of trainiing neural nets
train_imgs = train_imgs/255.0
test_imgs = test_imgs/255.0

#Defining mode - 3 layers:
#1st is input layer - flatten the images to a 1D array
#2nd is hidden layer (between input and output) of neurons that do the guessing 
# relu just returns positive value if negative one comes back
#3rd is output layer - 10 neurons for the 10 possible labels - each will have probability 
# that image matches to that layer and softmax picks the largest one
model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

#adding new metric will tell it to return the accuracy after each epoch and not just the loss
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#model.fit(train_imgs, train_labels, epochs=5)


#Can write a class to define when model training should stop instead of just inputting # of epochs
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.95):
      print("\nReached 95% accuracy so cancelling training!")
      self.model.stop_training = True
      
callback = myCallback()
model.fit(train_imgs, train_labels, epochs=50, callbacks=[callback])

model.evaluate(test_imgs, test_labels)

#using trained model to predict test data and showing comparison to actual value
classifications = model.predict(test_imgs)
print(classifications[0])
print(test_labels[0])

