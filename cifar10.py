import keras
import cv2
import os
import sys
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# IS_POSITION_BASED = True
IS_POSITION_BASED = int(sys.argv[1])
image_size = 32
batch_size = 128
image_channels = 3
if IS_POSITION_BASED:
    image_channels = 5
num_classes = 10
epochs = 500
data_augmentation = True

nx, ny = (32, 32)
xt = np.linspace(0, 1, nx)
yt = np.linspace(0, 1, ny)
xpos, ypos = np.meshgrid(xt, yt)

def xy_pos_add(images):
    processed_images = []
    count = 1
    for image in images:
	if count % 5000 == 0:
        	print "preprocessing image "+str(count) + "/" + str(len(images))
        image = np.reshape(image, (32, 32, 3))
        img = np.swapaxes(np.swapaxes(x_train[0], 0, 2), 1, 2)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if IS_POSITION_BASED:
            asd = np.swapaxes(np.swapaxes(np.array(np.concatenate([[img[0]],[img[1]], [img[2]], [xpos], [ypos]])), 0, 1), 1, 2)
        else:
            asd = image
        processed_images.append(asd)
        count = count + 1
    return np.array(processed_images)

print('x_train shape:', x_train.shape)
new_x_train = xy_pos_add(x_train)
new_x_test = xy_pos_add(x_test)

new_y_train = keras.utils.to_categorical(y_train, num_classes)
new_y_test = keras.utils.to_categorical(y_test, num_classes)

# normalize data
new_x_train = new_x_train.astype('float32')
new_x_test = new_x_test.astype('float32')
new_x_train /= 255
new_x_test /= 255

# model defination
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import CSVLogger

model_vgg16_conv = VGG16(weights="imagenet", include_top=False)
model_vgg16_conv.summary()

# graph setup
img_input = Input(shape=(image_size, image_size, image_channels), name="input_image")
output_vgg16_conv = model_vgg16_conv(img_input)

x = Flatten(name = 'flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='layer1')(x)
x = Dense(4096, activation = 'relu', name='layer2')(x)
x = Dense(1024, activation = 'relu', name='layer3')(x)
oput = Dense(10, activation = 'softmax', name='output')(x)

model = Model(input=img_input, output=oput)
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# create direct..
if not os.path.exists("..results"):
    os.mkdir("../results")
    os.mkdir("../results/position")
    os.mkdir("../results/position/best_models")
    os.mkdir("../results/normal")
    os.mkdir("../results/normal/best_models")

# checkpoints and logs

if IS_POSITION_BASED:
    checkpointer = ModelCheckpoint(filepath="../results/position/best_models/fn_model.{epoch:02d}-{val_acc:.6f}.hdf5", verbose=1, monitor='val_acc', save_best_only=True, save_weights_only=False, mode='max', period=1)
    tf_board = TensorBoard(log_dir='../results/position/logs', histogram_freq=0, write_graph=True, write_images=True)
    csv_logger = CSVLogger('../results/position/training.log')
else :
    checkpointer = ModelCheckpoint(filepath="../results/normal/best_models/fn_model.{epoch:02d}-{val_acc:.6f}.hdf5", verbose=1, monitor='val_acc', save_best_only=True, save_weights_only=False, mode='max', period=1)
    tf_board = TensorBoard(log_dir='../results/normal/logs', histogram_freq=0, write_graph=True, write_images=True)
    csv_logger = CSVLogger('../results/normal/training.log')

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# model fitting
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(new_x_train, new_y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.05,
              callbacks=[early_stopping, checkpointer, tf_board, csv_logger],
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(new_x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(new_x_train, new_y_train, batch_size=batch_size),
                        epochs=epochs,
                        validation_split = 0.05,
                        callbacks=[early_stopping, checkpointer, tf_board, csv_logger])


# testing network
if IS_POSITION_BASED:
    model.save("./results/position/best_models/final_model_vgg16.hdf5")
    score, acc = model.evaluate(new_x_test, new_y_test, batch_size=batch_size)
    resultsfile = open("./results/position/results.txt", 'w')
    resultsfile.write("test_acc: "+str(acc)+"\n")
    resultsfile.write("test_score: " + str(score))
    resultsfile.close()
else:
    model.save("./results/normal/best_models/final_model_vgg16.hdf5")
    score, acc = model.evaluate(new_x_test, new_y_test, batch_size=batch_size)
    resultsfile = open("./results/normal/results.txt", 'w')
    resultsfile.write("test_acc: "+str(acc)+ "\n")
    resultsfile.write("test_score: "+str(score))
    resultsfile.close()

                        
