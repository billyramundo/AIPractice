#%%
import tensorflow as tf
import tensorflow_datasets as tfds

#tfds makes it easy to load datasets of all kinds for model training
mnist_data = tfds.load("fashion_mnist")
#this will print the names of the diff groups in the dataset (test and train)
for i in mnist_data:
    print(i)
#this gives you the actual data in the given split
mnist_train = tfds.load(name="fashion_mnist", split='train')
#the take command allows you to look at individual entries in the data and get their diff parts
for i in mnist_train.take(1):
    print(type(i))
    print(i.keys())
    print(i['image'])
    print(i['label'])
    
#Can also get metadata/info on the dataset - use this flag
mnist_test, info = tfds.load(name="fashion_mnist", with_info="True")
# %%
#Using tfds with models - a few modifications required - as_supervised needed to get into this tuple format 
(training_images, training_labels), (test_images, test_labels) =  tfds.as_numpy(tfds.load('fashion_mnist',
                                                                                          split = ['train', 'test'], batch_size=-1, 
                                                                                          as_supervised=True))

training_images = training_images/255.0
test_images = test_images/255.0

#Shape is 28x28x1 already when comes from tfds so just need to specify this in input layer
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
# %%
#To do image augmentation with data foom tfds use a mapping function (diff than with DataGenerator from last chapter)
#tf.image library has a lot of useful augmentation functions
def augment_images(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/255)
    image = tf.image.random_flip_left_right(image)
    return image, label
    
data = tfds.load('horses_or_humans', split='train', as_supervised=True)
train = data.map(augment_images)
train_batches = train.shuffle(100).batch(32)
# %%
#Another useful tf package is tensorflow-addons - has a lot of functionality for helping train models
import tensorflow_addons as tfa

def augmentimages(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/255)
  image = tfa.image.rotate(image, 40, interpolation='NEAREST')
  return image, label
# %%
#tfds lets you split data how you want it - can specify the split in different ways

#first 10,000
data = tfds.load('cats_and_dogs', split='train[:10000]', as_supervised=True)
#first 20%
data2 = tfds.load('cats_and_dogs', split='train[:20%]', as_supervised=True)
#last 1000 and first 1000
data3 = data = tfds.load('cats_and_dogs', split='train[-1000:]+train[:1000]', as_supervised=True)

#this is the only way to count how many entries in the data you split and ensure you did it right
#is very slow so only use if need to debug
train_length = [i for i,_ in enumerate(data)][-1] + 1
print(train_length)
#%%
#ETL is the pattern used in tensorflow training - Extract, Transform, Load
#The way that the load phase is approached can greatly affect training speed
#Often perfom extract and transform with the CPU while loading is left for the GPU because its way more resource intensive
#Instead of doing these things in sequence you can really optimize training speed by doing them in parallel - 
# extract/transform the next batch of data while the previous one is being loaded - this process is called PIPELINING
#Pipelining allows very large datasets to be ETL'd relatively quickly, and tf provides APIs to enact it
import multiprocessing

train_data = tfds.load('cats_vs_dogs', split='train', with_info=True)

# %%
#Parallelizing the Extract phase
file_pattern = '../../../tensorflow_datasets/cats_vs_dogs/4.0.0/cats_vs_dogs-train.tfrecord*'
files = tf.data.Dataset.list_files(file_pattern)

#cycle_length is how many records are processed on each thread, and num_parallel_calls is how many threads 
#which is set dynamically here based on CPU availability
train_dataset = files.interleave(
                     tf.data.TFRecordDataset, 
                     cycle_length=4,
                     num_parallel_calls=tf.data.experimental.AUTOTUNE
                )

#Parallelizing the Transform phase
def read_tfrecord(serialized_example):
  feature_description={
      "image": tf.io.FixedLenFeature((), tf.string, ""),
      "label": tf.io.FixedLenFeature((), tf.int64, -1),
  }
  example = tf.io.parse_single_example(
       serialized_example, feature_description
  )
  image = tf.io.decode_jpeg(example['image'], channels=3)
  image = tf.cast(image, tf.float32)
  image = image / 255
  image = tf.image.resize(image, (300,300))
  return image, example['label']

#use multiprocessing package
cores = multiprocessing.cpu_count()
print(cores)
train_dataset = train_dataset.map(read_tfrecord, num_parallel_calls=cores)
#Can use this to speed things up but takes a lot of RAM so be careful
#train_dataset = train_dataset.cache()

#Parallelizing the load phase
train_dataset = train_dataset.shuffle(1024).batch(32)
#Tells the computer to prefetch based on number of cores available (again set dynamically)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
model.fit(train_dataset, epochs=10, verbose=1)
# %%
