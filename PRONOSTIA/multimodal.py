import tensorflow as tf
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, Conv1D, Dense, Concatenate, Conv2D, BatchNormalization,Activation,GlobalMaxPooling1D,LayerNormalization, MultiHeadAttention, Add, Reshape, Lambda, BatchNormalization, Activation, MaxPooling2D, Dense, Input, LSTM, MaxPooling1D, Concatenate
from keras._tf_keras.keras.regularizers import l2


reg_strength = 0.01


def image_branch(input_shape,reg_strength=0.01):

    visual_input = Input(shape=input_shape)

    # First Conv Block
    conv1 = Conv2D(32, (5, 5), padding='same', dilation_rate=(4,4), kernel_regularizer=l2(reg_strength))(visual_input)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = MaxPooling2D((3, 3), padding='same')(conv1)

    conv2 = Conv2D(32, (3, 3), padding='same', dilation_rate=(3,3), kernel_regularizer=l2(reg_strength))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = MaxPooling2D((3, 3),name='ires1ip', padding='same')(conv2)

    # Residual Connection (Fixed)
    res1 = Conv2D(32, (1, 1), padding='same', kernel_regularizer=l2(reg_strength))(visual_input) 
    res1 = Lambda(lambda x: tf.image.resize(x, (conv2.shape[1], conv2.shape[2])))(res1)  # Resize dynamically
    res1 = Add(name='ires1')([res1, conv2])

    # Second Conv Block
    conv3 = Conv2D(64, (3, 3), padding='same', dilation_rate=(2,2), kernel_regularizer=l2(reg_strength))(res1)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = MaxPooling2D((3, 3), padding='same')(conv3)

    conv4 = Conv2D(64, (2, 2), padding='same', dilation_rate=(1,1), kernel_regularizer=l2(reg_strength))(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = MaxPooling2D((2, 2),name='ires2ip', padding='same')(conv4)

    # Residual Connection (Fixed)
    res2 = Conv2D(64, (1, 1), padding='same', kernel_regularizer=l2(reg_strength))(res1)
    res2 = Lambda(lambda x: tf.image.resize(x, (conv4.shape[1], conv4.shape[2])))(res2)  # Resize dynamically
    res2 = Add(name='ires2')([res2, conv4])

    visual_out = Reshape((-1, int(res2.shape[-1])))(res2)  # (batch, seq_length, embed_dim)

    model = Model(inputs=visual_input, outputs=visual_out)
    return model


def time_branch(input_shape,reg_strength=0.01):

    time_series_input = Input(shape=input_shape)
    conv1 = Conv1D(32, 2, padding='same',dilation_rate=2, kernel_regularizer=l2(reg_strength))(time_series_input)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = MaxPooling1D(pool_size=1)(conv1)  

    conv2 = Conv1D(32, 2, padding='same',dilation_rate=2, kernel_regularizer=l2(reg_strength))(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    conv2 = MaxPooling1D(pool_size=1,name='tres1ip')(conv2)  

    res1 = Conv1D(32, 1, padding='same',activation='relu',kernel_regularizer=l2(reg_strength))(time_series_input)
    res1 = Add(name='tres1')([conv2,res1])

    # Second Conv Block with Residual Connection
    conv3 = Conv1D(64, 2, padding='same', dilation_rate=1, kernel_regularizer=l2(reg_strength))(res1)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    conv3 = MaxPooling1D(pool_size=1)(conv3)  # Adjust pooling size to maintain the same dimension

    conv4 = Conv1D(64, 2, padding='same', dilation_rate=1, kernel_regularizer=l2(reg_strength))(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = MaxPooling1D(pool_size=1,name='tres2ip')(conv4) 

    # Adjust res1 to match convh2 dimensions for residual connection
    res2 = Conv1D(64, 1, padding='same', activation='relu', kernel_regularizer=l2(reg_strength))(res1)
    time_out = Add(name='tres2')([res2, conv4])

    model = Model(inputs=time_series_input, outputs=time_out)
    return model


def multimod_rul(img_shape=(63,500,1),time_shape=(7,1),reg_strength=0.01):
    image_model = image_branch(input_shape=img_shape)
    time_model = time_branch(input_shape=time_shape)
    concat = Concatenate(axis=1,name='mres1ip')([time_model.output,image_model.output])

    # First LSTM Block with Residual Connection
    lstm1 = LSTM(100, activation='tanh', return_sequences=True)(concat)
    lstm2 = LSTM(64, activation='tanh', return_sequences=True)(lstm1)
    lstm2 = LSTM(64, activation='tanh', return_sequences=True)(lstm2)
    res = Add(name='mres1')([concat,lstm2])

    norm = LayerNormalization()(res)
    mha = MultiHeadAttention(num_heads=8, key_dim=64)(norm, norm,norm)
    out = GlobalMaxPooling1D()(mha)
    out = Dense(64, activation='relu',kernel_regularizer=l2(reg_strength))(out)
    out = Dense(32, activation='relu',kernel_regularizer=l2(reg_strength))(out)                                                       
    out = Dense(1)(out)
    model = Model(inputs=[image_model.input, time_model.input], outputs=out)
    return model


