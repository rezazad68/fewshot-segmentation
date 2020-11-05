## On the Texture Bias for Few-Shot CNN Segmentation, Implemented by Reza Azad ##
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import keras
import keras.layers as layers 
from keras.models import Model
import keras.backend as K

############################################ Encoder Weights on Image Net ###########################################
VGG_WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                           'releases/download/v0.1/'
                           'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

################################## VGG 16 Encoder #######################################
def vgg_encoder_b3(input_size = (256, 256, 3)):
    img_input = layers.Input(input_size) 
    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same', dilation_rate=(2, 2),
                      name='block3_conv3')(x)

    # Create model.
    model = Model(img_input, x, name='vgg16_model_with_block1-4')

    # Load weights.
    weights_path = keras.utils.get_file(
                   'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                   VGG_WEIGHTS_PATH_NO_TOP,
                   cache_subdir='models',
                   file_hash='6d6bbae143d832006294945121d1f1fc')
    model.load_weights(weights_path, by_name=True)
    
    return model


def vgg_encoder_b4(input_size = (256, 256, 3)):
    img_input = layers.Input(input_size) 
    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same', dilation_rate=(2, 2),
                      name='block3_conv3')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)

    # Create model.
    model = Model(img_input, x, name='vgg16_model_with_block1-4')

    # Load weights.
    weights_path = keras.utils.get_file(
                   'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                   VGG_WEIGHTS_PATH_NO_TOP,
                   cache_subdir='models',
                   file_hash='6d6bbae143d832006294945121d1f1fc')
    model.load_weights(weights_path, by_name=True)
    
    return model
    

def vgg_encoder_b34(input_size = (256, 256, 3)):
    img_input = layers.Input(input_size) 
    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x3 = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same', dilation_rate=(2, 2),
                      name='block3_conv3')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x3)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.concatenate([x, x3], axis=3) 

    # Create model.
    model = Model(img_input, x, name='vgg16_model_with_block1-4')

    # Load weights.
    weights_path = keras.utils.get_file(
                   'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                   VGG_WEIGHTS_PATH_NO_TOP,
                   cache_subdir='models',
                   file_hash='6d6bbae143d832006294945121d1f1fc')
    model.load_weights(weights_path, by_name=True)
    
    return model


def vgg_encoder_b5(input_size = (256, 256, 3)):
    img_input = layers.Input(input_size) 
    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same', dilation_rate=(2, 2),
                      name='block3_conv3')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same', dilation_rate=(2, 2),
                      name='block4_conv3')(x)
    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    
    # Create model.
    model = Model(img_input, x, name='vgg16_model_with_block1-5')

    # Load weights.
    weights_path = keras.utils.get_file(
                   'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                   VGG_WEIGHTS_PATH_NO_TOP,
                   cache_subdir='models',
                   file_hash='6d6bbae143d832006294945121d1f1fc')
    model.load_weights(weights_path, by_name=True)
    
    return model
    
def vgg_encoder_b345(input_size = (256, 256, 3)):
    img_input = layers.Input(input_size) 
    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x3 = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same', dilation_rate=(2, 2),
                      name='block3_conv3')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x3)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x4 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same', dilation_rate=(2, 2),
                      name='block4_conv3')(x)
    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x4)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
                      
    x = layers.concatenate([x, x4, x3], axis=3)
    # Create model.
    model = Model(img_input, x, name='vgg16_model_with_block1-5')

    # Load weights.
    weights_path = keras.utils.get_file(
                   'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                   VGG_WEIGHTS_PATH_NO_TOP,
                   cache_subdir='models',
                   file_hash='6d6bbae143d832006294945121d1f1fc')
    model.load_weights(weights_path, by_name=True)
    
    return model    

def vgg_encoder_b35(input_size = (256, 256, 3)):
    img_input = layers.Input(input_size) 
    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x3 = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same', dilation_rate=(2, 2),
                      name='block3_conv3')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x3)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x4 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same', dilation_rate=(2, 2),
                      name='block4_conv3')(x)
    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x4)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
                      
    x = layers.concatenate([x, x3], axis=3)
    # Create model.
    model = Model(img_input, x, name='vgg16_model_with_block1-5')

    # Load weights.
    weights_path = keras.utils.get_file(
                   'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                   VGG_WEIGHTS_PATH_NO_TOP,
                   cache_subdir='models',
                   file_hash='6d6bbae143d832006294945121d1f1fc')
    model.load_weights(weights_path, by_name=True)
    
    return model 
    
def vgg_encoder_b45(input_size = (256, 256, 3)):
    img_input = layers.Input(input_size) 
    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x3 = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same', dilation_rate=(2, 2),
                      name='block3_conv3')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x3)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x4 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same', dilation_rate=(2, 2),
                      name='block4_conv3')(x)
    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x4)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
                      
    x = layers.concatenate([x, x4], axis=3)
    # Create model.
    model = Model(img_input, x, name='vgg16_model_with_block1-5')

    # Load weights.
    weights_path = keras.utils.get_file(
                   'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                   VGG_WEIGHTS_PATH_NO_TOP,
                   cache_subdir='models',
                   file_hash='6d6bbae143d832006294945121d1f1fc')
    model.load_weights(weights_path, by_name=True)
    
    return model                 