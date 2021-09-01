import tensorflow as tf
import numpy as np
import pathlib,os,json,pickle
import IPython.display as display
import matplotlib.pyplot as plt
import pandas as pd

#==============Loading the model========

model = tf.keras.models.load_model('checkpoint/weightings.h5')#Load model structure and weightings
with open(os.path.join('checkpoint','label_to_index.txt'),'r') as file: #load the trained label_names
    label_names = json.loads(file.read())
label_names= {y:x for x,y in label_names.items()}#transform from label:number to number:label


#============Classify the category=============
#=======Creating image file reader============
def preprocess_image(image):#Input an image, resize the image, and preprocess it
     image = tf.image.decode_jpeg(image,channels=3)
     image = tf.image.crop_to_bounding_box(image,int(image.shape[0]*0.2),0,int(image.shape[0]*0.6),int(image.shape[1]))#cutting the left 20% and right 20% out
     image = tf.image.resize(image,[224,224])
     image/=255.0
     return image

def load_and_preprocess_image(path): #Input a path,read the image, and preprocess it
    image = tf.io.read_file(path)
    return preprocess_image(image)

#======Classification==========
testing_root = pathlib.Path('testing/')
all_testing_paths = [str(path) for path in list(testing_root.glob('*/*'))]

pred_list = []
for i in range(len(all_testing_paths)):
    img = load_and_preprocess_image(all_testing_paths[i])
    display.display(display.Image(all_testing_paths[i]))
    img = np.reshape(img,[1,224,224,3])
    pred_list.append(label_names[np.argmax(model.predict(img))])
    #test = enumerate(model.predict(img).tolist()[0])
    print(all_testing_paths[i])
    print(f'Lentil Detected: {pred_list[-1]}')
    print('*'*10)
    # if pred_list[-1] == 'not person':
    #     print('N/A')
    # else:
    #     print(pred_list[-1])

#==========Plotting accuracy/loss of trained model======
with open('checkpoint/history','rb') as file:
    previous_history = pickle.load(file)

acc = previous_history["accuracy"]
val_acc = previous_history["val_accuracy"]
loss = previous_history["loss"]
val_loss = previous_history["val_loss"]

plt.figure(figsize=(8,5))
plt.subplot(1,1,1)
plt.plot(acc,label='Training Accuracy')
plt.plot(val_acc,label='Validation Accuracy')
plt.legend(loc='lower right')
plt.suptitle('Performance of the neural network under new approach')
plt.xticks(np.arange(len(acc)),np.arange(1,len(acc)+1))
plt.xlabel('epoch')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
