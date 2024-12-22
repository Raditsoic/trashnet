import tensorflow as tf
import numpy as np
from src import augment_training_set

def unpack_and_resize_data(dataset, target_size=(224, 224)):
    images = []
    labels = []
    
    for item in dataset:
        img = tf.image.resize(item['image'], target_size)
        img = tf.keras.utils.img_to_array(img)
        images.append(img)
        labels.append(item['label'])
    
    return np.array(images), np.array(labels)

def prepare_dataset(images, labels, batch_size, is_training=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000, seed=42)
        dataset = dataset.map(
            lambda x, y: (augment_training_set(x), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        # Just normalize validation data
        dataset = dataset.map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


