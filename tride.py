import tensorflow as tf
print(tf.__version__)

from tensorflow.keras import Model, Input, Sequential
from tensorflow.keras.layers import Dense, Conv3D, UpSampling3D, concatenate, MaxPooling3D, BatchNormalization, Flatten, Dropout

shape = (384, 192,192, 1)
act = tf.nn.leaky_relu
model = Sequential([
	Conv3D(32, 3, activation=act, kernel_initializer='he_uniform'),
	MaxPooling3D(2),
	BatchNormalization(),
	Dropout(0.5),
	Conv3D(64, 3, activation=act, kernel_initializer='he_uniform'),
	MaxPooling3D(2),
	BatchNormalization(),
	Dropout(0.5),
	Flatten(),
	Dense(256, activation=act),
	Dense(1, activation='sigmoid')
])

def build_and_compile(model = model):
	model.compile(optimizer='rmsprop',loss = 'categorical_crossentropy', metrics=['accuracy'])
	model.build(shape)
	model.summary()

def Unet3D(shape,num_classes):
    inputs = Input(shape = shape)
    x = inputs
    conv1 = Conv3D(8, 3, activation = 'relu', padding = 'same',data_format="channels_last")(x)
    conv1 = Conv3D(8, 3, activation = 'relu', padding = 'same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)
    conv2 = Conv3D(16, 3, activation = 'relu', padding = 'same')(pool1)
    conv2 = Conv3D(16, 3, activation = 'relu', padding = 'same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = Conv3D(32, 3, activation = 'relu', padding = 'same')(pool2)
    conv3 = Conv3D(32, 3, activation = 'relu', padding = 'same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)
    conv4 = Conv3D(64, 3, activation = 'relu', padding = 'same')(pool3)
    conv4 = Conv3D(64, 3, activation = 'relu', padding = 'same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(drop4)

    conv5 = Conv3D(128, 3, activation = 'relu', padding = 'same')(pool4)
    conv5 = Conv3D(128, 3, activation = 'relu', padding = 'same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv3D(64, 2, activation = 'relu', padding = 'same')(UpSampling3D(size = (2,2,2))(drop5))
    merge6 = concatenate([drop4,up6],axis=-1)
    conv6 = Conv3D(64, 3, activation = 'relu', padding = 'same')(merge6)
    conv6 = Conv3D(64, 3, activation = 'relu', padding = 'same')(conv6)

    up7 = Conv3D(32, 2, activation = 'relu', padding = 'same')(UpSampling3D(size = (2,2,2))(conv6))
    merge7 = concatenate([conv3,up7],axis=-1)
    conv7 = Conv3D(32, 3, activation = 'relu', padding = 'same')(merge7)
    conv7 = Conv3D(32, 3, activation = 'relu', padding = 'same')(conv7)

    up8 = Conv3D(16, 2, activation = 'relu', padding = 'same')(UpSampling3D(size = (2,2,2))(conv7))
    merge8 = concatenate([conv2,up8],axis=-1)
    conv8 = Conv3D(16, 3, activation = 'relu', padding = 'same')(merge8)
    conv8 = Conv3D(16, 3, activation = 'relu', padding = 'same')(conv8)

    up9 = Conv3D(8, 2, activation = 'relu', padding = 'same')(UpSampling3D(size = (2,2,2))(conv8))
    merge9 = concatenate([conv1,up9],axis=-1)
    conv9 = Conv3D(8, 3, activation = 'relu', padding = 'same')(merge9)
    conv9 = Conv3D(8, 3, activation = 'relu', padding = 'same')(conv9)
    conv10 = Conv3D(1, 1, activation = 'sigmoid')(conv9)
    model = Model(inputs=inputs, outputs = conv10)
    return model

def DenseNet(shape = shape, act = act):
	model = Sequential([
		Dense(256, activation=act, input_shape = shape),
		BatchNormalization(),
		Dropout(0.2),
		Dense(128, activation=act),
		BatchNormalization(),
		Dropout(0.2),
		Dense(1, activation='sigmoid')
	])
	return model

def get_model(width=128, height=128, depth=64):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

model = DenseNet(shape)

build_and_compile(model= model)