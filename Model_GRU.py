import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import matplotlib.pyplot as plt
from pprint import pprint
import copy

train_dataset= tf.data.experimental.load('./content/train.tfrecord')
validation_dataset= tf.data.experimental.load('./content/val.tfrecord')
test_dataset= tf.data.experimental.load('./content/test.tfrecord')

SAMPLING_RATE = 2
SEQUENCE_LENGTH= 4
DELAY = SAMPLING_RATE * (SEQUENCE_LENGTH + 24 - 1)
BATCH_SIZE = 100

def model_builder(hp):
  inputs = keras.Input(shape=(SEQUENCE_LENGTH, X_train.shape[-1]))
  #Tuning units of LSTM layers
  hp_units = hp.Int('units', min_value=16, max_value=128, step=16)
  x = keras.layers.GRU(units=hp_units, recurrent_dropout=0.5, return_sequences=True, unroll=True)(inputs)
  
  hp_units1 = hp.Int('units', min_value=16, max_value=128, step=16)
  x = keras.layers.GRU(units=hp_units1, recurrent_dropout=0.5, unroll=True)(x)

  hp_dropout = hp.Choice('rate', values=[0.2,0.5,0.7])
  x = keras.layers.Dropout(rate=hp_dropout)(x)
  outputs = keras.layers.Dense(1)(x)
  GRUmodel = keras.Model(inputs, outputs)


  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2,  1e-4, 3e-4])
  GRUmodel.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                    loss='mse', metrics=['mse','mae',tf.keras.metrics.RootMeanSquaredError()])
  return GRUmodel


tuner = kt.Hyperband(model_builder,
                     objective='val_loss',
                     max_epochs=20,
                     factor=3,
                     directory='/content/',
                     project_name='GRU_tune_allusers_tt')

callbacks = [keras.callbacks.ModelCheckpoint('GRU_tune_allusers.keras', 
                                            save_best_only = True),
             keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=10)
             ]

tuner.search(train_dataset, epochs=10, validation_data=validation_dataset, callbacks=callbacks)

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')}, the best dropout is {best_hps.get('rate')}, and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
model = tuner.hypermodel.build(best_hps)
history = model.fit(train_dataset,
                    epochs=50,
                    validation_data=validation_dataset,
                    callbacks=callbacks)

val_loss_per_epoch = history.history['val_loss']
best_epoch = val_loss_per_epoch.index(min(val_loss_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)
callbacks = [keras.callbacks.ModelCheckpoint('GRU_tune_allusers.keras', 
                                            save_best_only = True)
             ]

# Retrain the model
tuneHistory = hypermodel.fit(train_dataset,
               epochs=100,
               validation_data=validation_dataset,
               callbacks=callbacks)

test_model = keras.models.load_model("/content/GRU_tune_allusers.keras")
test_loss= test_model.evaluate(test_dataset)
print(f"Test MSE: {test_loss}")

loss = tuneHistory.history['mse']
val_loss= tuneHistory.history['val_mse']
epochs = range(1, len(loss) +1) 
plt.figure()
plt.plot(epochs, loss, 'bo',label='Training MSE')
plt.plot(epochs, val_loss, 'b',label='Validation MSE')
plt.title("Training and Validation MAE")
plt.legend()
plt.show()