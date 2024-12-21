---
datasets:
- garythung/trashnet
metrics:
- accuracy
pipeline_tag: image-classification
---

# Model Version 1.2
Created on: 2024-12-21 21:58:40

```

## Metrics
- *train_loss*: 0.5705
- *train_accuracy*: 0.7812
- *val_loss*: 0.5975
- *val_accuracy*: 0.7752

## Configuration
- **architecture**: Tensorflow CNN
- **input_size**: (224, 224, 3)
- **num_classes**: 6
- **augmentation**: {'rotation_range': 20, 'width_shift_range': 0.2, 'height_shift_range': 0.2, 'horizontal_flip': True, 'vertical_flip': True, 'shear_range': 0.2, 'brightness_range': [0.8, 1.2]}
- **optimizer**: adam
- **learning_rate**: 0.001
- **batch_size**: 32
- **epochs**: 50
