import tensorflow as tf
import wandb
import os
import json
from datetime import datetime

class WandbMetricsLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        wandb.log({
            'epoch': epoch,
            'train_loss': logs.get('loss', 0),
            'train_accuracy': logs.get('accuracy', 0),
            'val_loss': logs.get('val_loss', 0),
            'val_accuracy': logs.get('val_accuracy', 0),
            'learning_rate': float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        })

def save_model(version, model, config, metrics, base_dir="../models"):
    model_dir = os.path.join(base_dir, f"model_{version}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model weights and architecture
    keras_model_path = os.path.join(model_dir, f"model_{version}.keras") 
    model.save(keras_model_path)
    
    
    # Save configs
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    
    # Save metrics
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    # Create README
    readme_path = os.path.join(model_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"""---
datasets:
- garythung/trashnet
metrics:
- accuracy
pipeline_tag: image-classification
---
""")
        f.write(f"\n# Model Version {version}\n")
        f.write(f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("```\n\n")
        f.write("## Metrics\n")
        for key, value in metrics.items():
            f.write(f"- *{key}*: {value:.4f}\n")
        
        f.write("\n## Configuration\n")
        for key, value in config.items():
            f.write(f"- **{key}**: {value}\n")
    
    print(f"Model version {version} saved successfully at {model_dir}.")
    return model_dir