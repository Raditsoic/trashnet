import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime

from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import wandb

class_names = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Trash']

def analyze_trashnet_model(model, history, X_val, y_val):
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Training & Validation Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='Training', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    plt.title('Model Accuracy over Epochs', fontsize=14, pad=20)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # 2. Training & Validation Loss
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Training', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
    plt.title('Model Loss over Epochs', fontsize=14, pad=20)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    
    # Get predictions
    y_val_classes = np.argmax(y_val, axis=1)
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # 3. Confusion Matrix
    plt.subplot(2, 2, 3)
    cm = confusion_matrix(y_val_classes, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=14, pad=20)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    
    # 4. Per-class Metrics
    plt.subplot(2, 2, 4)
    metrics_df = pd.DataFrame({
        'F1-Score': f1_score(y_val_classes, y_pred_classes, average=None),
        'Precision': precision_score(y_val_classes, y_pred_classes, average=None),
        'Recall': recall_score(y_val_classes, y_pred_classes, average=None)
    }, index=class_names)
    
    sns.barplot(data=metrics_df)
    plt.title('Per-class Metrics', fontsize=14, pad=20)
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout(pad=3.0)
    
    # Print detailed metrics
    print("\nClassification Report:")
    print(classification_report(y_val_classes, y_pred_classes, target_names=class_names))
    
    # Calculate and print overall metrics
    print("\nOverall Metrics:")
    print(f"F1-Score (weighted): {f1_score(y_val_classes, y_pred_classes, average='weighted'):.4f}")
    print(f"Precision (weighted): {precision_score(y_val_classes, y_pred_classes, average='weighted'):.4f}")
    print(f"Recall (weighted): {recall_score(y_val_classes, y_pred_classes, average='weighted'):.4f}")
    
    # Log metrics to wandb
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_val_classes,
            preds=y_pred_classes,
            class_names=class_names
        ),
        "f1_score": f1_score(y_val_classes, y_pred_classes, average='weighted'),
        "precision": precision_score(y_val_classes, y_pred_classes, average='weighted'),
        "recall": recall_score(y_val_classes, y_pred_classes, average='weighted')
    })
    
    return fig

def log_model_to_wandb(model, config, metrics, model_version):
    model_name = f"trashnet_model_v{model_version}"
    artifact = wandb.Artifact(
        name=model_name,
        type="model",
        description="Trashnet Classification model",
        metadata=config
    )
    
    # Save model locally first
    model_dir = "model_artifacts"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model in Keras format
    keras_model_path = os.path.join(model_dir, f"{model_name}.keras") 
    model.save(keras_model_path)
    
    # Save config
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    # Save metrics
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Create README with model info
    readme_path = os.path.join(model_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"# Trash Classification Model\n\n")
        f.write(f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Model Architecture\n```\n")

        f.write("```\n\n")
        f.write("## Metrics\n")
        for key, value in metrics.items():
            f.write(f"- {key}: {value}\n")
    
    # Add files to artifact
    artifact.add_dir(model_dir)
    
    # Log artifact to wandb
    wandb.log_artifact(artifact)
    
    return artifact