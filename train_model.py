# =============================================================
# AER 850 Project 2: Aircraft Defect Classification System
# Name: Jawad Rizwan
# Student Number: 501124033
# Due Date: November 5th, 2025
# -------------------------------------------------------------
# This script 
# =============================================================

# Suppress TensorFlow warnings 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

# Now import everything else
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

##########################################################
# STEP 1: Data Processing
##########################################################

print("="*60)
print("STEP 1: DATA PROCESSING")
print("="*60)

# 1. Define image parameters
IMG_HEIGHT = 500
IMG_WIDTH = 500
IMG_CHANNELS = 3
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
BATCH_SIZE = 32
NUM_CLASSES = 3  # crack, missing-head, paint-off

print(f"Image Shape: {IMG_SHAPE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Number of Classes: {NUM_CLASSES}")

# 2. Define data directories (using YOUR folder structure)
TRAIN_DIR = 'Data/train'
VALIDATION_DIR = 'Data/valid'
TEST_DIR = 'Data/test'

# Check if directories exist
print("\nChecking data directories...")
if not os.path.exists(TRAIN_DIR):
    print(f"❌ ERROR: {TRAIN_DIR} not found!")
    exit(1)
else:
    print(f"✓ Train directory found: {TRAIN_DIR}")
    
if not os.path.exists(VALIDATION_DIR):
    print(f"❌ ERROR: {VALIDATION_DIR} not found!")
    exit(1)
else:
    print(f"✓ Validation directory found: {VALIDATION_DIR}")

if not os.path.exists(TEST_DIR):
    print(f"❌ ERROR: {TEST_DIR} not found!")
    exit(1)
else:
    print(f"✓ Test directory found: {TEST_DIR}")

# 3. Data Augmentation for Training Data
print("\nSetting up data augmentation...")

# Training data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixel values to [0,1]
    shear_range=0.2,          # Shear transformation
    zoom_range=0.2,           # Random zoom
    horizontal_flip=True,     # Random horizontal flip
    rotation_range=20,        # Random rotation
    width_shift_range=0.2,    # Random horizontal shift
    height_shift_range=0.2    # Random vertical shift
)

# Validation data generator (only rescaling, no augmentation)
validation_datagen = ImageDataGenerator(
    rescale=1./255
)

print("✓ Data augmentation configured")
print("  - Training: rescaling, shear, zoom, flip, rotation, shifts")
print("  - Validation: rescaling only")

# 4. Create data generators
print("\nCreating data generators...")

try:
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',  # For multi-class classification
        shuffle=True,
        seed=42
    )
    print(f"✓ Training generator created: {train_generator.n} images found")
except Exception as e:
    print(f"❌ ERROR creating training generator: {e}")
    exit(1)

try:
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=42
    )
    print(f"✓ Validation generator created: {validation_generator.n} images found")
except Exception as e:
    print(f"❌ ERROR creating validation generator: {e}")
    exit(1)

# Print information about the data
print("\n" + "="*60)
print("DATA LOADING SUMMARY")
print("="*60)
print(f"Training samples: {train_generator.n}")
print(f"Validation samples: {validation_generator.n}")
print(f"Classes found: {train_generator.class_indices}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Steps per epoch (train): {train_generator.n // BATCH_SIZE}")
print(f"Steps per epoch (validation): {validation_generator.n // BATCH_SIZE}")
print("="*60)

# Verify we have the right number of samples (as per project specs)
expected_train = 1942
expected_val = 431

if train_generator.n == expected_train:
    print(f"✓ Training samples match expected: {expected_train}")
else:
    print(f"⚠️  Training samples ({train_generator.n}) don't match expected ({expected_train})")

if validation_generator.n == expected_val:
    print(f"✓ Validation samples match expected: {expected_val}")
else:
    print(f"⚠️  Validation samples ({validation_generator.n}) don't match expected ({expected_val})")

print("\n✓ STEP 1 COMPLETE: Data processing ready!")
print("="*60)

