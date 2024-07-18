import tensorflow as tf
from tensorflow.keras import layers
from keras.preprocessing import image

latent_dim = 100

# input label
con_label = layers.Input(shape=(1,))

# input latent vector 
latent_vector = layers.Input(shape=(latent_dim,))

def conditioned_label_generator(n_classes=3, embedding_dim=100):
    # embedding input
    embedding_label = layers.Embedding(n_classes, embedding_dim)(con_label)
    
    # linear multiplication
    no_nodes = 4 * 4 
    dense_label = layers.Dense(no_nodes)(embedding_label)
    
    # reshape to additional channel
    reshape_layer_label = layers.Reshape((4, 4, 1))(dense_label)
    return reshape_layer_label

def input_latent(latent_dim=100):
    # image generator input
    no_nodes = 512 * 4 * 4
    dense = layers.Dense(no_nodes)(latent_vector)
    dense = layers.ReLU()(dense)
    reshape_latent = layers.Reshape((4, 4, 512))(dense)
    return reshape_latent

def build_model():
    label_output = conditioned_label_generator()
    latent_vector_output= input_latent()
    
    # merge conditioned_label_generator and input_latent output
    merge = layers.Concatenate()([latent_vector_output, label_output])
    
    x = layers.Conv2DTranspose(64 * 8, kernel_size=4, strides= 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_1')(merge)
    
    x = layers.BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_1')(x)
    x = layers.ReLU(name='relu_1')(x)
    
    x = layers.Conv2DTranspose(64 * 4, kernel_size=4, strides= 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_2')(x)
    
    x = layers.BatchNormalization(momentum=0.1,  epsilon=0.8, center=1.0, scale=0.02, name='bn_2')(x)
    x = layers.ReLU(name='relu_2')(x)
    
    x = layers.Conv2DTranspose(64 * 2, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_3')(x)
    
    x = layers.BatchNormalization(momentum=0.1,  epsilon=0.8,  center=1.0, scale=0.02, name='bn_3')(x)
    x = layers.ReLU(name='relu_3')(x)
  

    x = layers.Conv2DTranspose(64 * 1, 4, 2, padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, name='conv_transpose_4')(x)
    
    x = layers.BatchNormalization(momentum=0.1,  epsilon=0.8,  center=1.0, scale=0.02, name='bn_4')(x)
    x = layers.ReLU(name='relu_4')(x) 
    
    out_layer = layers.Conv2DTranspose(3, 4, 2,padding='same', kernel_initializer=tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.02), use_bias=False, activation='tanh', name='conv_transpose_6')(x)
    
    model = tf.keras.Model([latent_vector,  con_label], out_layer)
    return model