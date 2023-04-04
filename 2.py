# YOUR CODE HERE
# raise NotImplementedError()
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

train_def = './ENG2006_Coursework_3_2023_Data/casting_data/train/def_front'
train_ok = './ENG2006_Coursework_3_2023_Data/casting_data/train/ok_front'
test_def = './ENG2006_Coursework_3_2023_Data/casting_data/test/def_front'
test_ok = './ENG2006_Coursework_3_2023_Data/casting_data/test/ok_front'

def load_images_from_folder(folder, label, size=(150, 150)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img_resized = cv2.resize(img, size)
            images.append(img_resized)
            labels.append(label)
    return images, labels

imagesTrainDef, labelsTrainDef = load_images_from_folder(train_def, 1)
imagesTrainOk, labelsTrainOk = load_images_from_folder(train_ok, 0)
imagesTestDef, labelsTestDef = load_images_from_folder(test_def, 1)
imagesTestOk, labelsTestOk = load_images_from_folder(test_ok, 0)

imagesTrain = imagesTrainDef + imagesTrainOk
imageLabelsTrain = labelsTrainDef + labelsTrainOk
imagesTest = imagesTestDef + imagesTestOk
imageLabelsTest = labelsTestDef + labelsTestOk

# Convert lists to numpy arrays
imagesTrain = np.array(imagesTrain)
imageLabelsTrain = np.array(imageLabelsTrain)
imagesTest = np.array(imagesTest)
imageLabelsTest = np.array(imageLabelsTest)


imagesTrain = imagesTrain / 255.0
imagesTest = imagesTest / 255.0
'''
# Plot one image from each set
plt.figure(figsize=(10, 10))

plt.subplot(1, 4, 1)
plt.imshow(imagesTrainDef[0], cmap='gray')
plt.title('Train Def')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(imagesTrainOk[0], cmap='gray')
plt.title('Train OK')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(imagesTestDef[0], cmap='gray')
plt.title('Test Def')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(imagesTestOk[0], cmap='gray')
plt.title('Test OK')
plt.axis('off')

plt.show()
'''
# imagesTrain, imageLabelsTrain = load_images_from_folder("./ENG2006_Coursework_3_2023_Data/casting_data/test/def_front")
# #imagesTest, imageLabelsTest = load_images_from_folder("path/to/test/folder")


#2-b

# YOUR CODE HERE
#raise NotImplementedError()
from sklearn.model_selection import train_test_split

# images_training, imagesVal, imagesLabel_training, imageLabelsVal = train_test_split(imagesTrain, imageLabelsTrain, test_size=0.75, random_state=42)
imagesTrain, imagesVal, imageLabelsTrain, imageLabelsVal = train_test_split(imagesTrain, imageLabelsTrain, test_size=0.25, random_state=42)


#2-c


# YOUR CODE HERE
#raise NotImplementedError()

import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Flatten

# Reshape the input data
imagesTrain = imagesTrain.reshape(imagesTrain.shape[0], 150, 150, 1)
imagesVal = imagesVal.reshape(imagesVal.shape[0], 150, 150, 1)

# Possible number of hidden layers
hidden_layers = [2, 4, 8]

# Possible number of hidden units
hidden_units = [32, 64, 128]

# Early stopping with a patience of 5
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

best_model = None
best_loss = float('inf')
best_acc = 0
best_layers = 0
best_units = 0

for layers in hidden_layers:
    for units in hidden_units:
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(Flatten(input_shape=(150, 150, 1)))
        
        # Add hidden layers with ReLU activation
        for _ in range(layers):
            model.add(tf.keras.layers.Dense(units, activation='relu'))
        
        # Add final dense layer with softmax activation
        model.add(tf.keras.layers.Dense(2, activation='softmax'))
        
        # Compile and fit the model for the training set
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics='accuracy')
        
        model.fit(imagesTrain, 
                  imageLabelsTrain,
                  validation_data=(imagesVal, imageLabelsVal),
                  epochs=100,
                  callbacks=[early_stopping],verbose=0)
        
        # Evaluate model for the validation set
        lossVal, accVal = model.evaluate(imagesVal, imageLabelsVal)
        print(f"{layers},{units}")# test
        # If the current model has a higher accuracy than the best model, update the best model
        if accVal > best_acc:
            best_model = model
            best_loss = lossVal
            best_acc = accVal
            best_layers = layers
            best_units = units

# Print details of the optimal model
print('Best loss over validation set: ', best_loss)
print('Best accuracy over validation set: ', best_acc)
print('Number of hidden layers selected: ', best_layers)
print('Number of hidden units selected: ', best_units)

# Save the optimal model
best_model.save('imageModelMLPOpt')





#DO NOT delete the following lines, they load your previously trained model
import tensorflow as tf
imageModelMLPOpt = tf.keras.models.load_model('imageModelMLPOpt')




imageMLPLossTest, imageMLPAccTest = best_model.evaluate(imagesTest, imageLabelsTest, verbose=0)


#Run this cell to test your answer for Question 2-c
from tests import question2c

question2c(imageModelMLPOpt,imageMLPLossTest,imageMLPAccTest,imagesTest,imageLabelsTest)