import tensorflow as tf
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, Conv1D, Dense, Concatenate, Conv2D, BatchNormalization,Activation,LayerNormalization, MultiHeadAttention, Add, Reshape, Lambda, MaxPooling2D, LSTM, MaxPooling1D,GlobalMaxPooling1D


reg_strength = 0.01

def mmv1():    
    reg_strength = 0.01 
    # Image Branch
    img_input = Input(shape=(63,500,1), name='image_input')
    img_conv1 = Conv2D(32, (5, 5), padding='same', kernel_regularizer=l2(reg_strength))(img_input)
    img_conv1 = BatchNormalization()(img_conv1)
    img_conv1 = Activation('relu')(img_conv1)
    img_conv1 = MaxPooling2D((3, 3), padding='same')(img_conv1)

    img_conv2 = Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(reg_strength))(img_conv1)
    img_conv2 = BatchNormalization()(img_conv2)
    img_conv2 = Activation('relu')(img_conv2)
    img_conv2 = MaxPooling2D((3, 3), padding='same')(img_conv2)

    img_res1 = Conv2D(32, (1, 1), padding='same', kernel_regularizer=l2(reg_strength))(img_input)
    img_res1 = Lambda(lambda x: tf.image.resize(x, (img_conv2.shape[1], img_conv2.shape[2])))(img_res1)
    img_res1 = Add()([img_res1, img_conv2])

    img_conv3 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(reg_strength))(img_res1)
    img_conv3 = BatchNormalization()(img_conv3)
    img_conv3 = Activation('relu')(img_conv3)
    img_conv3 = MaxPooling2D((3, 3), padding='same')(img_conv3)

    img_conv4 = Conv2D(64, (2, 2), padding='same', kernel_regularizer=l2(reg_strength))(img_conv3)
    img_conv4 = BatchNormalization()(img_conv4)
    img_conv4 = Activation('relu')(img_conv4)
    img_conv4 = MaxPooling2D((2, 2), padding='same')(img_conv4)

    img_res2 = Conv2D(64, (1, 1), padding='same', kernel_regularizer=l2(reg_strength))(img_res1)
    img_res2 = Lambda(lambda x: tf.image.resize(x, (img_conv4.shape[1], img_conv4.shape[2])))(img_res2)
    img_res2 = Add()([img_res2, img_conv4])

    img_output = Reshape((img_res2.shape[1] * img_res2.shape[2], img_res2.shape[3]))(img_res2)

    # Time-Series Branch
    time_input = Input(shape=(7, 1), name='time_input')
    time_conv1 = Conv1D(32, 2, padding='same', kernel_regularizer=l2(reg_strength))(time_input)
    time_conv1 = BatchNormalization()(time_conv1)
    time_conv1 = Activation('relu')(time_conv1)
    time_conv1 = MaxPooling1D(pool_size=1)(time_conv1)

    time_conv2 = Conv1D(32, 2, padding='same', kernel_regularizer=l2(reg_strength))(time_conv1)
    time_conv2 = BatchNormalization()(time_conv2)
    time_conv2 = Activation('relu')(time_conv2)
    time_conv2 = MaxPooling1D(pool_size=1)(time_conv2)

    time_res1 = Conv1D(32, 1, padding='same', activation='relu', kernel_regularizer=l2(reg_strength))(time_input)
    time_res1 = Add()([time_conv2, time_res1])

    time_conv3 = Conv1D(64, 2, padding='same', kernel_regularizer=l2(reg_strength))(time_res1)
    time_conv3 = BatchNormalization()(time_conv3)
    time_conv3 = Activation('relu')(time_conv3)
    time_conv3 = MaxPooling1D(pool_size=1)(time_conv3)

    time_conv4 = Conv1D(64, 2, padding='same', kernel_regularizer=l2(reg_strength))(time_conv3)
    time_conv4 = BatchNormalization()(time_conv4)
    time_conv4 = Activation('relu')(time_conv4)
    time_conv4 = MaxPooling1D(pool_size=1)(time_conv4)

    time_res2 = Conv1D(64, 1, padding='same', activation='relu', kernel_regularizer=l2(reg_strength))(time_res1)
    time_output = Add()([time_res2, time_conv4])

    # Fusion & Prediction
    concat_features = Concatenate(axis=1)([time_output, img_output])
    lstm1 = LSTM(100, activation='tanh', return_sequences=True)(concat_features)
    lstm2 = LSTM(64, activation='tanh', return_sequences=True)(lstm1)
    lstm2 = LSTM(64, activation='tanh', return_sequences=True)(lstm2)
    res_lstm = Add()([concat_features, lstm2])

    norm = LayerNormalization()(res_lstm)
    mha = MultiHeadAttention(num_heads=8, key_dim=64)(norm, norm, norm)
    output = GlobalMaxPooling1D()(mha)
    output = Dense(64, activation='relu', kernel_regularizer=l2(reg_strength))(output)
    output = Dense(32, activation='relu', kernel_regularizer=l2(reg_strength))(output)
    output = Dense(1, name='rul_output')(output)

    model = Model(inputs=[img_input, time_input], outputs=output)
    return model



def mmv2():
    reg_strength = 0.01
        
    # Image Branch
    img_input = Input(shape=(63,500,1), name='image_input')
    img_conv1 = Conv2D(32, (5, 5), padding='same', dilation_rate=(4,4), kernel_regularizer=l2(reg_strength))(img_input)
    img_conv1 = BatchNormalization()(img_conv1)
    img_conv1 = Activation('relu')(img_conv1)
    img_conv1 = MaxPooling2D((3, 3), padding='same')(img_conv1)

    img_conv2 = Conv2D(32, (3, 3), padding='same', dilation_rate=(3,3), kernel_regularizer=l2(reg_strength))(img_conv1)
    img_conv2 = BatchNormalization()(img_conv2)
    img_conv2 = Activation('relu')(img_conv2)
    img_conv2 = MaxPooling2D((3, 3), padding='same')(img_conv2)


    img_conv3 = Conv2D(64, (3, 3), padding='same', dilation_rate=(2,2), kernel_regularizer=l2(reg_strength))(img_conv2)
    img_conv3 = BatchNormalization()(img_conv3)
    img_conv3 = Activation('relu')(img_conv3)
    img_conv3 = MaxPooling2D((3, 3), padding='same')(img_conv3)

    img_conv4 = Conv2D(64, (2, 2), padding='same', dilation_rate=(1,1), kernel_regularizer=l2(reg_strength))(img_conv3)
    img_conv4 = BatchNormalization()(img_conv4)
    img_conv4 = Activation('relu')(img_conv4)
    img_conv4 = MaxPooling2D((2, 2), padding='same')(img_conv4)


    img_output = Reshape((img_conv4.shape[1] * img_conv4.shape[2], img_conv4.shape[3]))(img_conv4)

    # Time-Series Branch
    time_input = Input(shape=(7, 1), name='time_input')
    time_conv1 = Conv1D(32, 2, padding='same', dilation_rate=2, kernel_regularizer=l2(reg_strength))(time_input)
    time_conv1 = BatchNormalization()(time_conv1)
    time_conv1 = Activation('relu')(time_conv1)
    time_conv1 = MaxPooling1D(pool_size=1)(time_conv1)

    time_conv2 = Conv1D(32, 2, padding='same', dilation_rate=2, kernel_regularizer=l2(reg_strength))(time_conv1)
    time_conv2 = BatchNormalization()(time_conv2)
    time_conv2 = Activation('relu')(time_conv2)
    time_conv2 = MaxPooling1D(pool_size=1)(time_conv2)


    time_conv3 = Conv1D(64, 2, padding='same', dilation_rate=1, kernel_regularizer=l2(reg_strength))(time_conv2)
    time_conv3 = BatchNormalization()(time_conv3)
    time_conv3 = Activation('relu')(time_conv3)
    time_conv3 = MaxPooling1D(pool_size=1)(time_conv3)

    time_conv4 = Conv1D(64, 2, padding='same', dilation_rate=1, kernel_regularizer=l2(reg_strength))(time_conv3)
    time_conv4 = BatchNormalization()(time_conv4)
    time_conv4 = Activation('relu')(time_conv4)
    time_output = MaxPooling1D(pool_size=1)(time_conv4)


    # Fusion & Prediction
    concat_features = Concatenate(axis=1)([time_output, img_output])
    lstm1 = LSTM(100, activation='tanh', return_sequences=True)(concat_features)
    lstm2 = LSTM(64, activation='tanh', return_sequences=True)(lstm1)
    lstm2 = LSTM(64, activation='tanh', return_sequences=True)(lstm2)

    norm = LayerNormalization()(lstm2)
    mha = MultiHeadAttention(num_heads=8, key_dim=64)(norm, norm, norm)
    output = GlobalMaxPooling1D()(mha)
    output = Dense(64, activation='relu', kernel_regularizer=l2(reg_strength))(output)
    output = Dense(32, activation='relu', kernel_regularizer=l2(reg_strength))(output)
    output = Dense(1, name='rul_output')(output)

    model = Model(inputs=[img_input, time_input], outputs=output)
    return model



def mmv3():    
    # Image Branch
    img_input = Input(shape=(63,500,1), name='image_input')
    img_conv1 = Conv2D(32, (5, 5), padding='same', dilation_rate=(4,4), kernel_regularizer=l2(reg_strength))(img_input)
    img_conv1 = BatchNormalization()(img_conv1)
    img_conv1 = Activation('relu')(img_conv1)
    img_conv1 = MaxPooling2D((3, 3), padding='same')(img_conv1)

    img_conv2 = Conv2D(32, (3, 3), padding='same', dilation_rate=(3,3), kernel_regularizer=l2(reg_strength))(img_conv1)
    img_conv2 = BatchNormalization()(img_conv2)
    img_conv2 = Activation('relu')(img_conv2)
    img_conv2 = MaxPooling2D((3, 3), padding='same')(img_conv2)

    img_res1 = Conv2D(32, (1, 1), padding='same', kernel_regularizer=l2(reg_strength))(img_input)
    img_res1 = Lambda(lambda x: tf.image.resize(x, (img_conv2.shape[1], img_conv2.shape[2])))(img_res1)
    img_res1 = Add()([img_res1, img_conv2])

    img_conv3 = Conv2D(64, (3, 3), padding='same', dilation_rate=(2,2), kernel_regularizer=l2(reg_strength))(img_res1)
    img_conv3 = BatchNormalization()(img_conv3)
    img_conv3 = Activation('relu')(img_conv3)
    img_conv3 = MaxPooling2D((3, 3), padding='same')(img_conv3)

    img_conv4 = Conv2D(64, (2, 2), padding='same', dilation_rate=(1,1), kernel_regularizer=l2(reg_strength))(img_conv3)
    img_conv4 = BatchNormalization()(img_conv4)
    img_conv4 = Activation('relu')(img_conv4)
    img_conv4 = MaxPooling2D((2, 2), padding='same')(img_conv4)

    img_res2 = Conv2D(64, (1, 1), padding='same', kernel_regularizer=l2(reg_strength))(img_res1)
    img_res2 = Lambda(lambda x: tf.image.resize(x, (img_conv4.shape[1], img_conv4.shape[2])))(img_res2)
    img_res2 = Add()([img_res2, img_conv4])

    img_output = Reshape((img_res2.shape[1] * img_res2.shape[2], img_res2.shape[3]))(img_res2)

    # Time-Series Branch
    time_input = Input(shape=(7, 1), name='time_input')
    time_conv1 = Conv1D(32, 2, padding='same', dilation_rate=2, kernel_regularizer=l2(reg_strength))(time_input)
    time_conv1 = BatchNormalization()(time_conv1)
    time_conv1 = Activation('relu')(time_conv1)
    time_conv1 = MaxPooling1D(pool_size=1)(time_conv1)

    time_conv2 = Conv1D(32, 2, padding='same', dilation_rate=2, kernel_regularizer=l2(reg_strength))(time_conv1)
    time_conv2 = BatchNormalization()(time_conv2)
    time_conv2 = Activation('relu')(time_conv2)
    time_conv2 = MaxPooling1D(pool_size=1)(time_conv2)

    time_res1 = Conv1D(32, 1, padding='same', activation='relu', kernel_regularizer=l2(reg_strength))(time_input)
    time_res1 = Add()([time_conv2, time_res1])

    time_conv3 = Conv1D(64, 2, padding='same', dilation_rate=1, kernel_regularizer=l2(reg_strength))(time_res1)
    time_conv3 = BatchNormalization()(time_conv3)
    time_conv3 = Activation('relu')(time_conv3)
    time_conv3 = MaxPooling1D(pool_size=1)(time_conv3)

    time_conv4 = Conv1D(64, 2, padding='same', dilation_rate=1, kernel_regularizer=l2(reg_strength))(time_conv3)
    time_conv4 = BatchNormalization()(time_conv4)
    time_conv4 = Activation('relu')(time_conv4)
    time_conv4 = MaxPooling1D(pool_size=1)(time_conv4)

    time_res2 = Conv1D(64, 1, padding='same', activation='relu', kernel_regularizer=l2(reg_strength))(time_res1)
    time_output = Add()([time_res2, time_conv4])

    # Fusion & Prediction
    concat_features = Concatenate(axis=1)([time_output, img_output])
    lstm1 = LSTM(100, activation='tanh', return_sequences=True)(concat_features)
    lstm2 = LSTM(64, activation='tanh', return_sequences=True)(lstm1)
    lstm2 = LSTM(64, activation='tanh', return_sequences=True)(lstm2)
    res_lstm = Add()([concat_features, lstm2])

    output = GlobalMaxPooling1D()(res_lstm)
    output = Dense(64, activation='relu', kernel_regularizer=l2(reg_strength))(output)
    output = Dense(32, activation='relu', kernel_regularizer=l2(reg_strength))(output)
    output = Dense(1, name='rul_output')(output)

    model = Model(inputs=[img_input, time_input], outputs=output)
    return model