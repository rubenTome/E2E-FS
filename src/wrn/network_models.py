from keras.models import Model
from keras import backend as K, optimizers, layers, models
from keras.layers import Dense, Activation, BatchNormalization, Input, Convolution2D, GlobalAveragePooling2D, Flatten
from keras.regularizers import l2
from keras.applications import DenseNet121, MobileNetV2
from src.wrn.wide_residual_network import wrn_block
from src.network_models import three_layer_nn as tln
from src.layers.dfs import DFS
import numpy as np
import tempfile
import os
import keras
from backend_config import bcknd
# import tensorflow_model_optimization as tfmot

K.backend = bcknd


def three_layer_nn(input_shape, nclasses=2, bn=True, kernel_initializer='he_normal',
                   dropout=0.0, dfs=False, regularization=5e-4, momentum=0.9):

    nfeatures = np.prod(input_shape)
    tln_model = tln((nfeatures, ), nclasses, bn, kernel_initializer, dropout, False, regularization, momentum)
    ip = Input(shape=input_shape)
    x = ip
    if dfs:
        x = DFS()(x)
    if len(input_shape) > 1:
        x = Flatten()(x)
    output = tln_model(x)
    model = Model(ip, output)

    optimizer = optimizers.SGD(learning_rate=1e-1)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model

def three_layer_nn_q(input_shape, nclasses=2, bn=True, kernel_initializer='he_normal',
                   dropout=0.0, dfs=False, regularization=5e-4, layer_dims=None, quantized=False, momentum=0.9):

    layersL = []
    if dfs:
        layersL.append(DFS())
    else:
        layersL.append(layers.Flatten())

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    if layer_dims is None:
        layer_dims = [150, 100, 50]
    if dfs:
        layersL.append(DFS())
    for layer_dim in layer_dims:
        layersL.append(layers.Dense(layer_dim, use_bias=not bn, kernel_initializer=kernel_initializer,
                  kernel_regularizer=l2(regularization) if regularization > 0.0 else None))
        if bn:
            layersL.append(layers.BatchNormalization(axis=channel_axis, momentum=momentum, epsilon=1e-5, gamma_initializer='ones'))
        if dropout > 0.0:
            layers.append(layers.Dropout(dropout))
        layersL.append(layers.Activation('relu'))

    layersL.append(layers.Dense(nclasses, use_bias=True, kernel_initializer=kernel_initializer,
              kernel_regularizer=l2(regularization) if regularization > 0.0 else None))
    layersL.append(layers.Activation('softmax'))

    model = keras.Sequential(layersL)
    #cuantizamos modelo
    if quantized:
        model = tfmot.quantization.keras.quantize_model(model)
    #TODO se usan 2 optimizadores distintos en three_layer_nn
    #optimizer = optimizers.Adam(lr=1e-4)
    optimizer = optimizers.SGD(learning_rate=1e-1)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model

def wrn164(
    input_shape, nclasses=2, bn=True, kernel_initializer='he_normal', dropout=0.0, dfs=False, regularization=0.0,
    softmax=True
):

    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    ip = Input(shape=input_shape)

    x = ip
    if dfs:
        x = DFS()(x)

    x = Convolution2D(
        16, (3, 3), padding='same', kernel_initializer=kernel_initializer,
        use_bias=False, kernel_regularizer=l2(regularization) if regularization > 0.0 else None
    )(x)

    l = 16
    k = 4

    output_channel_basis = [16, 32, 64]
    strides = [1, 2, 2]

    N = (l - 4) // 6

    for ocb, stride in zip(output_channel_basis, strides):
        x = wrn_block(
            x, ocb * k, N, strides=(stride, stride), dropout=dropout,
            regularization=regularization, kernel_initializer=kernel_initializer, bn=bn
        )

    if bn:
        x = BatchNormalization(axis=channel_axis, momentum=0.9, epsilon=1e-5, gamma_initializer='ones')(x)
    x = Activation('relu')(x)

    deep_features = GlobalAveragePooling2D()(x)

    classifier = Dense(nclasses, kernel_initializer=kernel_initializer)

    output = classifier(deep_features)
    if softmax:
        output = Activation('softmax')(output)

    model = Model(ip, output)

    optimizer = optimizers.SGD(learning_rate=1e-1)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    return model

def densenet(
        input_shape, nclasses=2, num_dense_blocks=3, growth_rate=12, depth=100, compression_factor=0.5,
        data_augmentation=True, regularization=None, dfs=False
):

    num_bottleneck_layers = (depth - 4) // (2 * num_dense_blocks)
    num_filters_bef_dense_block = 2 * growth_rate

    # start model definition
    # densenet CNNs (composite function) are made of BN-ReLU-Conv2D
    inputs = layers.Input(shape=input_shape)
    x = inputs
    if dfs:
        x = DFS()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(num_filters_bef_dense_block,
                      kernel_size=3, kernel_regularizer=l2(regularization) if regularization > 0.0 else None,
                      padding='same',
                      kernel_initializer='he_normal')(x)
    x = layers.concatenate([inputs, x])

    # stack of dense blocks bridged by transition layers
    for i in range(num_dense_blocks):
        # a dense block is a stack of bottleneck layers
        for j in range(num_bottleneck_layers):
            y = layers.BatchNormalization()(x)
            y = layers.Activation('relu')(y)
            y = layers.Conv2D(4 * growth_rate,
                              kernel_size=1, kernel_regularizer=l2(regularization) if regularization > 0.0 else None,
                              padding='same',
                              kernel_initializer='he_normal')(y)
            if not data_augmentation:
                y = layers.Dropout(0.2)(y)
            y = layers.BatchNormalization()(y)
            y = layers.Activation('relu')(y)
            y = layers.Conv2D(growth_rate,
                              kernel_size=3, kernel_regularizer=l2(regularization) if regularization > 0.0 else None,
                              padding='same',
                              kernel_initializer='he_normal')(y)
            if not data_augmentation:
                y = layers.Dropout(0.2)(y)
            x = layers.concatenate([x, y])

        # no transition layer after the last dense block
        if i == num_dense_blocks - 1:
            continue

        # transition layer compresses num of feature maps and reduces the size by 2
        num_filters_bef_dense_block += num_bottleneck_layers * growth_rate
        num_filters_bef_dense_block = int(num_filters_bef_dense_block * compression_factor)
        y = layers.BatchNormalization()(x)
        y = layers.Conv2D(num_filters_bef_dense_block,
                          kernel_size=1, kernel_regularizer=l2(regularization) if regularization > 0.0 else None,
                          padding='same',
                          kernel_initializer='he_normal')(y)
        if not data_augmentation:
            y = layers.Dropout(0.2)(y)
        x = layers.AveragePooling2D()(y)

    # add classifier on top
    # after average pooling, size of feature map is 1 x 1
    x = layers.AveragePooling2D()(x)
    y = layers.Flatten()(x)
    outputs = layers.Dense(nclasses,
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2(regularization) if regularization > 0.0 else None,
                           activation='softmax')(y)

    # instantiate and compile model
    # orig paper uses SGD but RMSprop works better for DenseNet
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(1e-3),
                  metrics=['acc'])

    return model


def efficientnetB0(
        input_shape, nclasses=2, num_dense_blocks=3, growth_rate=12, depth=100, compression_factor=0.5,
        data_augmentation=True, regularization=0., dfs=False
):

    keras_shape = input_shape
    if input_shape[-1] == 1:
        keras_shape = (32, 32, 3)

    keras_model = EfficientNetB0(
        include_top=False,
        input_shape=keras_shape,
        weights=None
    )

    keras_model.trainable = True

    # adding regularization
    # regularizer = l2(regularization)
    #
    # for layer in keras_model.layers:
    #     for attr in ['kernel_regularizer']:
    #         if hasattr(layer, attr):
    #             setattr(layer, attr, regularizer)
    #
    # tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    # keras_model.save_weights(tmp_weights_path)
    #
    # keras_json = keras_model.to_json()
    # keras_model = models.model_from_json(keras_json)
    # keras_model.load_weights(tmp_weights_path, by_name=True)

    outputs = keras_model.output
    inputs = keras_model.input
    if input_shape[-1] == 1 or dfs:
        inputs = layers.Input(shape=input_shape)
        x = inputs
        if dfs:
            x = DFS()(x)
        if input_shape[-1] == 1:
            x = layers.ZeroPadding2D(padding=(2, 2))(x)
            output_shape = K.int_shape(x)
            output_shape = output_shape[:-1] + (3,)
            x = layers.Lambda(lambda x: K.tile(x, (1, 1, 1, 3)), output_shape=output_shape)(x)
        outputs = keras_model(x)

    outputs = layers.Flatten()(outputs)
    # outputs = layers.GlobalAveragePooling2D()(outputs)
    # outputs = layers.Dropout(rate=.2)(outputs)
    outputs = layers.Dense(nclasses,
                           kernel_initializer='he_normal',
                           kernel_regularizer=None,
                           activation='softmax')(outputs)

    # instantiate and compile model
    # orig paper uses SGD but RMSprop works better for DenseNet
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(1e-4),
                  metrics=['acc'])

    return model


def efficientnetB1(
        input_shape, nclasses=2, num_dense_blocks=3, growth_rate=12, depth=100, compression_factor=0.5,
        data_augmentation=True, regularization=0.
):

    keras_shape = input_shape
    if input_shape[-1] == 1:
        keras_shape = (32, 32, 3)

    keras_model = EfficientNetB1(
        include_top=False,
        input_shape=keras_shape,
        weights=None
    )

    keras_model.trainable = True

    # adding regularization
    # regularizer = l2(regularization)
    #
    # for layer in keras_model.layers:
    #     for attr in ['kernel_regularizer']:
    #         if hasattr(layer, attr):
    #             setattr(layer, attr, regularizer)
    #
    # tmp_weights_path = os.path.join(tempfile.gettempdir(), 'tmp_weights.h5')
    # keras_model.save_weights(tmp_weights_path)
    #
    # keras_json = keras_model.to_json()
    # keras_model = models.model_from_json(keras_json)
    # keras_model.load_weights(tmp_weights_path, by_name=True)

    outputs = keras_model.output
    inputs = keras_model.input
    if input_shape[-1] == 1:
        inputs = layers.Input(shape=input_shape)
        x = layers.ZeroPadding2D(padding=(2, 2))(inputs)
        output_shape = K.int_shape(x)
        output_shape = output_shape[:-1] + (3,)
        x = layers.Lambda(lambda x: K.tile(x, (1, 1, 1, 3)), output_shape=output_shape)(x)
        outputs = keras_model(x)

    # outputs = layers.Flatten()(outputs)
    outputs = layers.GlobalAveragePooling2D()(outputs)
    # outputs = layers.Dropout(rate=.2)(outputs)
    outputs = layers.Dense(nclasses,
                           kernel_initializer='he_normal',
                           kernel_regularizer=None,
                           activation='softmax')(outputs)

    # instantiate and compile model
    # orig paper uses SGD but RMSprop works better for DenseNet
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(1e-4),
                  metrics=['acc'])

    return model

#
# def efficientnetB0v2(
#         input_shape, nclasses=2, num_dense_blocks=3, growth_rate=12, depth=100, compression_factor=0.5,
#         data_augmentation=True, regularization=0.
# ):
#     model_shape = (224, 224, 3)
#
#     keras_model = EfficientNetB0(
#         include_top=False,
#         input_shape=model_shape,
#         weights=None
#     )
#
#     keras_model.trainable = True
#
#     inputs = layers.Input(shape=input_shape)
#     x = layers.experimental.preprocessing.Resizing(model_shape[0], model_shape[1], interpolation='bicubic')(inputs)
#     #x = layers.experimental.preprocessing.RandomTranslation(height_factor=4./32., width_factor=4./32.)(x)
#     x = RandomElasticDeform(displacement=np.array([2,3,3]))(x)
#
#
#     if input_shape[-1] == 1:
#         output_shape = K.int_shape(x)
#         output_shape = output_shape[:-1] + (3,)
#         x = layers.Lambda(lambda x: K.tile(x, (1, 1, 1, 3)), output_shape=output_shape)(x)
#
#     outputs = keras_model(x)
#     outputs = layers.GlobalAveragePooling2D()(outputs)
#     outputs = layers.Dropout(rate=.5)(outputs)
#     outputs = layers.Dense(nclasses,
#                            kernel_initializer='he_normal',
#                            # kernel_regularizer=l2(regularization) if regularization > 0.0 else None,
#                            activation='softmax')(outputs)
#
#     # instantiate and compile model
#     # orig paper uses SGD but RMSprop works better for DenseNet
#     model = models.Model(inputs=inputs, outputs=outputs)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=optimizers.Adam(1e-4),
#                   metrics=['acc'], run_eagerly=True)
#
#     return model
#
