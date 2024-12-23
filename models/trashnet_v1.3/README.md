---
datasets:
- garythung/trashnet
metrics:
- accuracy
pipeline_tag: image-classification
---

# Model Version 1.3
Created on: 2024-12-22 15:00:20

```

## Metrics
- *train_loss*: 0.7903
- *train_accuracy*: 0.7188
- *val_loss*: 0.6419
- *val_accuracy*: 0.7671

## Configuration
- **architecture**: 3 Convolutional Layers, 2 Dense Layers, Global Average Pooling, Dropout 0.3
- **input_size**: (224, 224, 3)
- **num_classes**: 6
- **augmentation**: {'rotation_range': 20, 'width_shift_range': 0.2, 'height_shift_range': 0.2, 'horizontal_flip': True, 'vertical_flip': True, 'shear_range': 0.2, 'brightness_range': [0.8, 1.2]}
- **optimizer**: adam
- **learning_rate**: 0.001
- **batch_size**: 32
- **epochs**: 50
