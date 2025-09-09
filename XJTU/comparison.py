import tensorflow as tf
import numpy as np
from keras._tf_keras.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras._tf_keras.keras.models import Model,Sequential
from keras._tf_keras.keras.layers import Input, Conv1D, GRU, Dense, Concatenate, Dropout, Softmax, GlobalAveragePooling1D, Conv2D, GlobalAveragePooling2D, Multiply,BatchNormalization,Activation,Bidirectional,LeakyReLU,GlobalMaxPooling1D,Attention,concatenate
from keras._tf_keras.keras.regularizers import l2
from sklearn.model_selection import KFold
from keras._tf_keras.keras.optimizers import Adam
import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import math
import pandas as pd
import math
import tensorflow as tf
import tqdm  # For progress bar
import tqdm  # For progress bar
import tensorflow as tf
import os
import tensorflow as tf
from keras._tf_keras.keras.layers import LayerNormalization, MultiHeadAttention, Add, Reshape, Lambda, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Input, LSTM, Conv1D, MaxPooling1D, Concatenate,Resizing
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import RMSprop
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle


np.random.seed(100)

input_shape = (14, 1)
reg_strength = 0.005


def CNN_LSTM():
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape,padding='same'))  # 1D Conv Layer
    model.add(MaxPooling1D(2))  # Max pooling layer
    model.add(Conv1D(128, 3, activation='relu',padding='same'))  # 2nd Conv Layer
    model.add(MaxPooling1D(2))  # Max pooling layer
    model.add(LSTM(100, activation='relu', return_sequences=False))  # LSTM layer
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))  # Output layer (for regression)
    model.compile(optimizer=RMSprop(learning_rate=1e-5), loss='mse', metrics=['mae'])
    return model


def CNN_BiLSTM():
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape,padding='same'))  # 1D Conv Layer
    model.add(MaxPooling1D(2))  # Max pooling layer
    model.add(Conv1D(128, 3, activation='relu',padding='same'))  # 2nd Conv Layer
    model.add(MaxPooling1D(2))  # Max pooling layer
    model.add(Bidirectional(LSTM(100, activation='relu', return_sequences=False)))  # BILSTM layer
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))  # Output layer (for regression)
    model.compile(optimizer=RMSprop(learning_rate=1e-5), loss='mse', metrics=['mae'])
    return model


def MSIDISN():
    ms_input = Input(shape=input_shape)
    conv1 = Conv1D(64, 3, activation='relu', padding='same')(ms_input)
    conv2 = Conv1D(64, 5, activation='relu', padding='same')(ms_input)
    conv3 = Conv1D(64, 7, activation='relu', padding='same')(ms_input)
    concatenated = concatenate([conv1, conv2, conv3],axis=-1)
    max_pool = MaxPooling1D(2)(concatenated)
    attention_output = Attention()([max_pool, max_pool])
    bgru = Bidirectional(GRU(100, activation='relu', return_sequences=True))(attention_output)
    lstm = LSTM(100, activation='relu', return_sequences=False)(bgru)
    dense1 = Dense(50, activation='relu')(lstm)
    output_layer = Dense(1)(dense1)  # Output layer (RUL prediction)
    model = Model(inputs=ms_input, outputs=output_layer)
    model.compile(optimizer=RMSprop(learning_rate=1e-5), loss='mse', metrics=['mae'])
    return model



def CARLE():

    convH = Input(shape=input_shape)
    convh1 = Conv1D(64, 3, padding='same', kernel_regularizer=l2(reg_strength))(convH)
    convh1 = BatchNormalization()(convh1)
    convh1 = Activation('relu')(convh1)
    convh1 = MaxPooling1D(pool_size=1)(convh1)  # Adjust pooling size to maintain the same dimension
    convH_adjusted = Conv1D(64, 1, padding='same', kernel_regularizer=l2(reg_strength))(convH)
    res1 = Add()([convH_adjusted, convh1])
    convh2 = Conv1D(64, 3, padding='same', kernel_regularizer=l2(reg_strength))(res1)
    convh2 = BatchNormalization()(convh2)
    convh2 = Activation('relu')(convh2)
    convh2 = MaxPooling1D(pool_size=1)(convh2)  # Adjust pooling size to maintain the same dimension
    res1_adjusted = Conv1D(64, 1, padding='same', kernel_regularizer=l2(reg_strength))(res1)
    res2 = Add()([res1_adjusted, convh2])
    convh3 = Conv1D(64, 2, padding='same', kernel_regularizer=l2(reg_strength))(res2)
    convh3 = BatchNormalization()(convh3)
    convh3 = Activation('relu')(convh3)
    convh3 = MaxPooling1D(pool_size=1)(convh3)  # Adjust pooling size to maintain the same dimension
    res2_adjusted = Conv1D(64, 1, padding='same', kernel_regularizer=l2(reg_strength))(res2)
    res3 = Add()([res2_adjusted, convh3])
    convh4 = Conv1D(64, 2, padding='same', kernel_regularizer=l2(reg_strength))(res3)
    convh4 = BatchNormalization()(convh4)
    convh4 = Activation('relu')(convh4)
    convh4 = MaxPooling1D(pool_size=1)(convh4)  # Adjust pooling size to maintain the same dimension
    res3_adjusted = Conv1D(64, 1, padding='same', kernel_regularizer=l2(reg_strength))(res3)
    res4 = Add()([res3_adjusted, convh4])
    multi_head_attention_h = MultiHeadAttention(num_heads=8, key_dim=64)(res4, res4)
    lstm1 = LSTM(64, activation='relu', stateful=False, return_sequences=True)(multi_head_attention_h)
    multi_head_attention_h = MultiHeadAttention(num_heads=8, key_dim=64)(lstm1, lstm1)
    res5 = Add()([multi_head_attention_h, lstm1])
    lstm2 = LSTM(64, activation='relu', stateful=False, return_sequences=True)(res5)
    multi_head_attention_h = MultiHeadAttention(num_heads=8, key_dim=64)(lstm2, lstm2)
    res6 = Add()([res5, multi_head_attention_h])
    flat = Flatten()(res6)
    dense = Dense(64, activation='relu', kernel_regularizer=l2(reg_strength))(flat)
    dense = Dense(48, activation='relu', kernel_regularizer=l2(reg_strength))(dense)
    dense = Dense(32, activation='relu', kernel_regularizer=l2(reg_strength))(dense)
    out = Dense(1)(dense)
    model = Model(inputs=convH, outputs=out)
    return model