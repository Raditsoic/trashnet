import tensorflow as tf

def augment_training_set(data):
    if tf.reduce_max(data) <= 1.0:
        data = data * 255.0
    
    data_augmentation = tf.keras.Sequential([
        # Flips
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomTranslation(0.2, 0.2),
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        
        # Color and intensity adjustments
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
        
        # Normalize last
        tf.keras.layers.Rescaling(1./255)
    ])
    
    augmented_data = data_augmentation(data)
    
    augmented_data = tf.clip_by_value(augmented_data, 0.0, 1.0)
    
    return augmented_data

