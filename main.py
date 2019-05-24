import matplotlib.pyplot as plt
import os
import numpy as np

from keras.callbacks import ModelCheckpoint, EarlyStopping
from SegNet import *
from load_data import *
from keras.utils import multi_gpu_model

X_train, y_train, X_test, Y_test = load_data()

num_classes = 4
Y_train = to_categorical(y_train, num_classes)
model = SegNet()
prallel_model = multi_gpu_model(model, gpus=2)
prallel_model.compile(
    loss="categorical_crossentropy",
    optimizer='adam',
    metrics=['accuracy'])
prallel_model.summary()
# model_checkpoint = ModelCheckpoint('Weights.h5',
#                                    monitor='val_loss',
#                                    save_best_only=True)
early_stopping = EarlyStopping(patience=10,
                               mode='auto',
                               monitor='val_loss',
                               restore_best_weights=True,
                               verbose=1)
print('Fitting model...')
history = prallel_model.fit(X_train, Y_train,
                            batch_size=50,
                            epochs=10,
                            validation_split=0.2,
                            shuffle=True,
                            callbacks=[early_stopping],
                            # validation_data=(X_test, Y_test),
                            verbose=1)
train_score = prallel_model.evaluate(X_train, Y_train, verbose=1)
print('train_score', train_score)
Y_test = to_categorical(Y_test, num_classes)
verify_score = prallel_model.evaluate(X_test, Y_test, verbose=1)
print('verify_score', verify_score)
# prallel_model.load_weights('Weights.h5')
Y_test = prallel_model.predict(X_test, verbose=1)
Y_test = Y_test.reshape(20, 88, 88, 4)
Y = np.argmax(Y_test, axis=-1)

print('Saving predicted masks to files...')

pred_dir = 'Predictions'
if not os.path.exists(pred_dir):
    os.mkdir(pred_dir)
for i in range(20):
    image = Y[i, :, : ]
    plt.imsave(os.path.join(pred_dir, str(i + 1) + '_pred.png'), image, cmap = 'viridis')