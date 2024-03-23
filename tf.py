import keras
import tensorflow as tf


print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))
