import numpy as np
import os, sys
import sklearn.metrics as metrics
import wide_residual_network as wrn
from keras.datasets import cifar10
import keras.callbacks as callbacks, ModelCheckpoint, TensorBoard, EarlyStopping, CSVLogger
import keras.utils.np_utils as kutils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras import backend as K
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


batch_size = 64
nb_epoch = 100
img_rows, img_cols = 32, 32

nx, ny = (32, 32)
xt = np.linspace(0, 1, nx)
yt = np.linspace(0, 1, ny)
xpos, ypos = np.meshgrid(xt, yt)

IS_POSITION_BASED = int(sys.argv[1])

# def xy_pos_add(images):
#     processed_images = []
#     count = 1
#     for image in images:
# 	if count % 5000 == 0:
#         	print "preprocessing image "+str(count) + "/" + str(len(images))
#         image = np.reshape(image, (3, 32, 32))
#         if IS_POSITION_BASED:
#             asd = np.swapaxes(np.swapaxes(np.array(np.concatenate([[image[0]], [image[1]], [image[2]], [xpos], [ypos]])), 0, 1), 1, 2)
#         else:
#             asd = np.swapaxes(np.swapaxes(np.array(image), 0, 1), 1, 2)
#         processed_images.append(asd)
#         count = count + 1
#     return np.array(processed_images)


(trainX, trainY), (testX, testY) = cifar10.load_data()
print('x_train shape:', x_train.shape)
if IS_POSITION_BASED:
    trainX = np.swapaxes(np.swapaxes(trainX, 0, 3), 1, 3)
    trainX = np.swapaxes(np.swapaxes(np.swapaxes(np.vstack([[trainX[0]], [trainX[1]], [trainX[2]], [[xpos,]*trainX.shape[1]], [[ypos,]*trainX.shape[1]]]), 0, 2), 0, 1), 2, 3)
    testX = np.swapaxes(np.swapaxes(testX, 0, 3), 1, 3)
    testX = np.swapaxes(np.swapaxes(np.swapaxes(np.vstack([[testX[0]], [testX[1]], [testX[2]], [[xpos,]*testX.shape[1]], [[ypos,]*testX.shape[1]]]), 0, 2), 0, 1), 2, 3)
    

trainX = trainX.astype('float32')
trainX /= 255.0
testX = testX.astype('float32')
testX /= 255.0

tempY = testY
trainY = kutils.to_categorical(trainY)
testY = kutils.to_categorical(testY)

generator = ImageDataGenerator(featurewise_center=True,
                               featurewise_std_normalization=True,
                               rotation_range=10,
                               width_shift_range=5./32,
                               height_shift_range=5./32,
                               horizontal_flip=True)

generator.fit(trainX, seed=0, augment=False)

test_generator = ImageDataGenerator(featurewise_center=True,
                                    featurewise_std_normalization=True,)

test_generator.fit(testX, seed=0, augment=False)

init_shape = (3, 32, 32) if K.image_dim_ordering() == 'th' else (32, 32, 3)

# For WRN-16-8 put N = 2, k = 8
# For WRN-28-10 put N = 4, k = 10
# For WRN-40-4 put N = 6, k = 4
model = wrn.create_wide_residual_network(init_shape, nb_classes=10, N=2, k=8, dropout=0.0)

model.summary()
#plot_model(model, "WRN-28-8.png", show_shapes=False)

model.compile(loss="categorical_crossentropy", optimizer="adadelta", metrics=["acc"])
print("Finished compiling") 
print("Allocating GPU memory")

# model.load_weights("weights/WRN-28-8 Weights.h5")
# print("Model loaded.")

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


model.fit_generator(generator.flow(trainX, trainY, batch_size=batch_size), steps_per_epoch=len(trainX) // batch_size + 1, nb_epoch=nb_epoch,
                   callbacks=[early_stopping, checkpointer, tf_board, csv_logger],
                   validation_data=test_generator.flow(testX, testY, batch_size=batch_size),
                   validation_steps=testX.shape[0] // batch_size + 1,)

if IS_POSITION_BASED:
    model.save("../results/position/best_models/final_model_vgg16.hdf5")
    score, acc = model.evaluate_generator(test_generator.flow(testX, testY, nb_epoch), testX.shape[0] // batch_size + 1)
    resultsfile = open("../results/position/results.txt", 'w')
    resultsfile.write("test_acc: "+str(acc)+"\n")
    resultsfile.write("test_score: " + str(score))
    resultsfile.close()
else:
    model.save("../results/normal/best_models/final_model_vgg16.hdf5")
    score, acc = model.evaluate_generator(test_generator.flow(testX, testY, nb_epoch), testX.shape[0] // batch_size + 1)
    resultsfile = open("../results/normal/results.txt", 'w')
    resultsfile.write("test_acc: "+str(acc)+ "\n")
    resultsfile.write("test_score: "+str(score))
    resultsfile.close()