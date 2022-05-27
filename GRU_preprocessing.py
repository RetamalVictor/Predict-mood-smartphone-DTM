import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pprint import pprint
import copy
pd.set_option('max_columns', None)

pd.set_option('max_rows', None)


#Loading and preparing the data with the right columns 
df = pd.read_csv('/content/drive/MyDrive/DTM_advanced/Datasets/data_all_prepro.csv')
#df = pd.read_csv('/content/data_all_prepro.csv')
df.drop('Unnamed: 0', axis=1,inplace=True)

#Placing mood at the end to make it easier
cols = list(df)[:]
cols.pop(cols.index('mood')) 
df = df[cols+['mood']]
mood = copy.copy(df[['id','mood']].to_numpy())
mood = pd.DataFrame(mood, columns=['id','mood'])
#Standardizing
scaler = StandardScaler()
floats = df.drop(['id','Date'], axis=1)
#Fit on training
scaler = scaler.fit(floats.values)
#transform
data_scaled = scaler.transform(floats.values)
scaled_df = pd.DataFrame(data_scaled, columns =floats.columns)
scaled_df['id'] = df['id']

#Uncomment for One hot Encoding
# df_one_hot = pd.get_dummies(scaled_df.id, prefix='user')
# df_one = pd.concat([scaled_df,df_one_hot],axis=1)
# print(f"df_one_train shape: {df_one.shape}")

#Creating Tensorflow Datasets
train_df = pd.DataFrame()
val_df = pd.DataFrame()
test_df = pd.DataFrame()
train_y = pd.DataFrame()
val_y = pd.DataFrame()
test_y = pd.DataFrame()
users = df['id'].unique()
user_test = users[len(users)-1:]
users = users[:len(users)-1]


#Split in Train Validation & Test
for user in users:
  #new_df = df_one[df_one['id'] == user]
  new_df = scaled_df[scaled_df['id'] == user]
  new_y = mood[mood['id']==user]
  num_train =int(new_df.shape[0]*0.8)
  num_val= int(new_df.shape[0]*0.2)
  num_test = len(new_df) - (num_train+num_val) 

  train_df = pd.concat([train_df, new_df[:num_train]], axis=0)
  val_df = pd.concat([val_df, new_df[num_train:num_train+num_val]], axis=0)
  test_df = pd.concat([test_df, new_df[num_train+num_val:]], axis=0)
  
  train_y = pd.concat([train_y, new_y[:num_train]], axis=0)
  val_y = pd.concat([val_y, new_y[num_train:num_train+num_val]], axis=0)
  test_y = pd.concat([test_y, new_y[num_train+num_val:]], axis=0)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df= train_df.drop('id',axis=1)
val_df= val_df.drop('id',axis=1)
test_df= test_df.drop('id',axis=1)
y_train = train_y['mood'].to_numpy()
y_val = val_y['mood'].to_numpy()
y_test = test_y['mood'].to_numpy()

print(f"train_df shape: {train_df.shape}")
print(f"val_df shape: {val_df.shape}")
print(f"test_df shape: {test_df.shape}")

train_df=train_df.astype(float) 
val_df=val_df.astype(float) 
test_df=test_df.astype(float) 
X_train,X_val, X_test = train_df.values, val_df.values, test_df.values

y_train = np.asarray(y_train).astype('float32')
y_val = np.asarray(y_val).astype('float32')
y_test = np.asarray(y_test).astype('float32')
#All values are prepared to be transformed into TF dataset


# Tensorflow Dataset with window sliding
# https://www.tensorflow.org/tutorials/structured_data/time_series#data_windowing
SAMPLING_RATE = 2
SEQUENCE_LENGTH= 4
DELAY = SAMPLING_RATE * (SEQUENCE_LENGTH + 24 - 1)
BATCH_SIZE = 100

train_dataset = keras.utils.timeseries_dataset_from_array(
  X_train,
  targets=y_train,
  sampling_rate=SAMPLING_RATE,
  sequence_length=SEQUENCE_LENGTH,
  shuffle=True,
  batch_size=BATCH_SIZE )

validation_dataset = keras.utils.timeseries_dataset_from_array(
  X_val,
  targets=y_val,
  sampling_rate=SAMPLING_RATE,
  sequence_length=SEQUENCE_LENGTH,
  shuffle=True,
  batch_size=BATCH_SIZE )

test_dataset = keras.utils.timeseries_dataset_from_array(
  X_test,
  targets=y_test,
  sampling_rate=SAMPLING_RATE,
  sequence_length=SEQUENCE_LENGTH,
  shuffle=True,
  batch_size=BATCH_SIZE )

for samples, targets in train_dataset:
  print(f"Sample shape: {samples.shape}")
  print(f"Targets shape: {targets.shape}")
  break 

tf.data.experimental.save(train_dataset,'.content/train')
tf.data.experimental.save(validation_dataset,'.content/val')
tf.data.experimental.save(test_dataset,'.content/test')
