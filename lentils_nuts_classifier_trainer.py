import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib,random,os,json,pickle

#============Data Preprocessing===============

#Getting all image paths
data_root = pathlib.Path('image/')#All training images should be stored under image folder->category name folder-->image file
all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)
img_count = len(all_image_paths)


#Getting the names of all categories and label them
label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())#Get all folders name under image/
label_to_index = dict((name,index) for index,name in enumerate(label_names))#Assign a number label to all categories
all_image_labels = [label_to_index[pathlib.Path(path).parent.name] for path in all_image_paths]# create a list of labels matching the list sequence of all all_image_paths

#Creating image file reader
def preprocess_image(image):#Input an image, resize the image, and preprocess it
     image = tf.image.decode_jpeg(image,channels=3)
     image = tf.image.resize(image,[224,224])
     image/=255.0
     return image

def load_and_preprocess_image(path): #Input a path,read the image, and preprocess it
    image = tf.io.read_file(path)
    return preprocess_image(image)


#Saving the labels for the trained model
with open(os.path.join('checkpoint','label_to_index.txt'),'w') as file:
    file.write(json.dumps(label_to_index))

# Train Test Split
datalen = len(all_image_paths)*0.8
train_ds = tf.data.Dataset.from_tensor_slices((all_image_paths[0:int(datalen*0.8)],all_image_labels[0:int(datalen*0.8)]))
val_ds = tf.data.Dataset.from_tensor_slices((all_image_paths[int(datalen*0.8):int(datalen)],all_image_labels[int(datalen*0.8):int(datalen)]))

def load_and_preprocess_from_path_label(path,label): #The tuples are unpacked into positional arguments of the mapped function
    return load_and_preprocess_image(path),label


train_label_ds = train_ds.map(load_and_preprocess_from_path_label)#Transform (path,label) to (image,label)
val_label_ds = val_ds.map(load_and_preprocess_from_path_label)


#Making the model to fetch random "BATCh_SIZE" images from the training dataset, in demand =======
BATCH_SIZE = 32

AUTOTUNE = tf.data.experimental.AUTOTUNE #Adjust the use of hardware automatically

train_ds = train_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=int(img_count*0.8))) #Shuffle the dataset, repeat/copy the dataset infinitely , on demand
train_ds = train_ds.batch(BATCH_SIZE)# Cut the shuffled, infinitely repeated dataset into batch of size "BATCH_SIZE"
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE) #Allow data preprocessing and model training at the same time


val_ds = val_label_ds.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=int(img_count*0.2))) #Shuffle the dataset, repeat/copy the dataset infinitely , on demand
val_ds = val_ds.batch(BATCH_SIZE)# Cut the shuffled, infinitely repeated dataset into batch of size "BATCH_SIZE"
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)


#================TRAINING==================

#=======Model Construction=========
inception_net = tf.keras.applications.InceptionV3(input_shape=(224,224,3),weights='imagenet',include_top=False)
inception_net.trainable=False
model = tf.keras.Sequential([
    inception_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(202,activation='relu'),
    tf.keras.layers.Dense(202,activation='relu'),
    tf.keras.layers.Dense(len(label_names),activation='softmax')
])

model.compile(optimizer=tf.compat.v1.train.RMSPropOptimizer(learning_rate=0.0001),
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=["accuracy"])


#=========== Setting checkpoints for training results saving=============
checkpoint_path = 'checkpoint/cp-{epoch:04d}.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,verbose=1,save_weights_only=True,period=20)


#========Training the model===========
steps_per_epoch = tf.math.ceil(int(len(all_image_paths)*0.8)/BATCH_SIZE).numpy() #Set the steps per epoch according to BATCH_SIZE
val_step = tf.math.ceil(int(len(all_image_paths)*0.2)/BATCH_SIZE).numpy()

history = model.fit(train_ds,
                    epochs=10,
                    steps_per_epoch=steps_per_epoch,
                    callbacks = [cp_callback],
                    validation_data = val_ds,
                    validation_steps=val_step)

#========= Saving the weightings history===========
tf.keras.models.save_model(model,'checkpoint/weightings.h5')
with open('checkpoint/history','wb') as file:
    pickle.dump(history.history,file)

#==========Plotting the results==============
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.figure(figsize=(8,8))
plt.subplot(2,1,1)
plt.plot(acc,label='Training Accuracy')
plt.plot(val_acc,label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2,1,2)
plot.plot(loss,label='Training Loss')
plt.plot(val_loss,label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

'''
## ========Continue previous training if u want to train again===================
model = tf.keras.models.load_moel('checkpoint/weightings.h5')#Load model structure
model.compile(optimizer=tf.compat.v1.train.RMSPropOptimizer(learning_rate=0.000001),
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=["accuracy"])


#=========== Setting checkpoints for training results saving=============
checkpoint_path = 'checkpoint/cp-{epoch:04d}.ckpt'
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,verbose=1,save_weights_only=True,period=20)


#========Training the model===========
steps_per_epoch = tf.math.ceil(int(len(all_image_paths)*0.8)/BATCH_SIZE).numpy() #Set the steps per epoch according to BATCH_SIZE
val_step = tf.math.ceil(int(len(all_image_paths)*0.2)/BATCH_SIZE).numpy()

history = model.fit(train_ds,
                    epochs=18,
                    initial_epoch = 15,
                    steps_per_epoch=steps_per_epoch,
                    callbacks = [cp_callback],
                    validation_data = val_ds,
                    validation_steps=val_step)


#========= Saving the weightings history===========
tf.keras.models.save_model(model,'checkpoint/weightings.h5')
with open('checkpoint/history','rb') as file:
    previous_history = pickle.load(file)


history_combine = {}
for i in previous_history:
    history_combine[i] = previous_history[i] + history.history[i]


acc = history_combine["accuracy"]
val_acc = history_combine["val_accuracy"]
loss = history_combine["loss"]
val_loss = history_combine["val_loss"]

plt.figure(figsize=(8,8))
plt.subplot(2,1,1)
plt.plot(acc,label='Training Accuracy')
plt.plot(val_acc,label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2,1,2)
plot.plot(loss,label='Training Loss')
plt.plot(val_loss,label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,max(plt.ylim())])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()'''
