import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import unpack_and_resize_data, augment_training_set, prepare_dataset, analyze_trashnet_model, log_model_to_wandb, save_model, WandbMetricsLogger, TrashnetTFModel

import numpy as np
import tensorflow as tf

from datasets import load_dataset
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import argparse
import wandb

def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def train(epochs=50, batch_size=32, learning_rate=1e-4, wandb_project="trashnet", model_name="CPU-3 Convolutions, 2 Dense Layers, Dropout 0.3", model_version="1.3"):
    set_seeds(50)

    dataset = load_dataset("garythung/trashnet")
    X, y = unpack_and_resize_data(dataset['train'], target_size=(224, 224))

    num_classes = len(np.unique(y))
    y = to_categorical(y, num_classes)
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Augment the training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        shear_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    ).flow(X_train, y_train, batch_size=batch_size)
    val_datagen = ImageDataGenerator(rescale=1./255).flow(X_val, y_val, batch_size=batch_size)

    wandb.init(
        project=wandb_project,  
        config={
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "architecture": {
                "conv_layers": 4,
                "dense_layers": 2,
                "flat layer": "Global Average Pool",
            },
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "dataset": "trashnet",
            "num_classes": num_classes,
        },
        name=model_name,
    )

    # Create and compile model
    model = TrashnetTFModel(input_shape=(224, 224, 3), num_classes=num_classes).build_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=wandb.config.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=wandb.config.loss,
        metrics=['accuracy']    
    )

    
    # train_generator = train_datagen.flow(
    #     X_train, y_train,
    #     batch_size=wandb.config.batch_size,
    #     shuffle=True
    # )

    # val_generator = val_datagen.flow(
    #     X_val, y_val,
    #     batch_size=wandb.config.batch_size,
    #     shuffle=False
    # )

    # Define callbacks
    callbacks = [
        WandbMetricsLogger(),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    # Train model
    history = model.fit(
        train_datagen,
        epochs=wandb.config.epochs,
        validation_data=val_datagen,
        callbacks=callbacks
    )

    analyze_trashnet_model(model, history, X_val, y_val)

    final_metrics = {
        'final_train_loss': history.history['loss'][-1],
        'final_train_accuracy': history.history['accuracy'][-1],
        'final_val_loss': history.history['val_loss'][-1],
        'final_val_accuracy': history.history['val_accuracy'][-1],
        'total_epochs': len(history.history['loss'])
    }

    log_model_to_wandb(
        model,
        wandb.config.as_dict(),
        final_metrics,
        model_version
    )

    wandb.finish()

    config = {
    "architecture": "3 Convolutional Layers, 2 Dense Layers, Global Average Pooling, Dropout 0.3",
    "input_size": (224, 224, 3),
    "num_classes": num_classes,
    "augmentation": {
        "RandomRotation(0.1)",
        "RandomTranslation(0.2, 0.2)",
        "RandomFlip('horizontal_and_vertical')",
        "RandomBrightness(0.2)",
        "RandomContrast(0.2)",
        "Rescaling(1./255)"
        },
    "optimizer": "adam",
    "learning_rate": wandb.config.learning_rate,
    "batch_size": batch_size,
    "epochs": wandb.config.epochs
    }
    
    final_metrics = {
        "train_loss": history.history['loss'][-1],
        "train_accuracy": history.history['accuracy'][-1],
        "val_loss": history.history['val_loss'][-1],
        "val_accuracy": history.history['val_accuracy'][-1]
    }

    save_model(model_version, model, config, final_metrics)
    
    return history, model



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Push TensorFlow model to HuggingFace Hub')
    parser.add_argument('--epochs', required=False, default=50, help='Training epochs')
    parser.add_argument('--batch_size', required=False, default=32, help='Batch size')
    parser.add_argument('--learning_rate', required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--wandb_project', required=False, default="trashnet", help='Wandb project name')
    parser.add_argument('--model_name', required=False, default="CPU-3 Convolutions, 2 Dense Layers, Dropout 0.3", help='Model name')
    parser.add_argument('--model_version', required=False, default="1.3", help='Model version')
    
    args = parser.parse_args()

    history, model = train(args.epochs, args.batch_size, args.learning_rate, args.wandb_project, args.model_name, args.model_version)