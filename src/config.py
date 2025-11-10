S = 7        # Grid size
B = 2        # Number of bounding boxes per grid cell
C = 20       # Number of classes (for PASCAL VOC)

IMAGE_SIZE = 448   # Input image size (448x448)
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 135

LAMBDA_COORD = 5.0  # Weight for coordinate loss
LAMBDA_NOOBJ = 0.5  # Weight for confidence loss of cells without objects

DROPOUT_RATE = 0.5
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

# Data augmentation parameters
SCALE_RANGE = 0.2         # Random scaling ±20%
TRANSLATION_RANGE = 0.2   # Random translation ±20%
HSV_ADJUST = 1.5          # Factor for random exposure & saturation adjustment
