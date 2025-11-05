# =============================================================
# AER 850 Project 2: Aircraft Defect Classification System (Steps 1-4)
# Name: Jawad Rizwan
# Student Number: 501124033
# Due Date: November 5th, 2025
# -------------------------------------------------------------
# This script 
# =============================================================

# Import os and suppress TensorFlow warnings 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

# Import necessary libraries
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau  

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

# 2. Define data directories 
TRAIN_DIR = 'Data/train'
VALIDATION_DIR = 'Data/valid'
TEST_DIR = 'Data/test'

# Check if directories exist
print("\nChecking data directories...")
if not os.path.exists(TRAIN_DIR):
    print(f"ERROR: {TRAIN_DIR} not found!")
    exit(1)
else:
    print(f"Train directory found: {TRAIN_DIR}")
    
if not os.path.exists(VALIDATION_DIR):
    print(f"ERROR: {VALIDATION_DIR} not found!")
    exit(1)
else:
    print(f"Validation directory found: {VALIDATION_DIR}")

if not os.path.exists(TEST_DIR):
    print(f"ERROR: {TEST_DIR} not found!")
    exit(1)
else:
    print(f"Test directory found: {TEST_DIR}")

# 3. Data Augmentation for Training Data
print("\nSetting up data augmentation...")

# Training data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,          
    zoom_range=0.1,           
    horizontal_flip=True,      
    rotation_range=10,         
    width_shift_range=0.2,    
    height_shift_range=0.2,   
    fill_mode='nearest'
)

# Validation data generator (only rescaling, no augmentation)
validation_datagen = ImageDataGenerator(
    rescale=1./255
)

print("Data augmentation configured")
print("  - Training: rescaling, shear, zoom, flips, rotation, shifts, brightness")
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
    print(f"Training generator created: {train_generator.n} images found")
except Exception as e:
    print(f"ERROR creating training generator: {e}")
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
    print(f"ERROR creating validation generator: {e}")
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
    print(f"Training samples match expected: {expected_train}")
else:
    print(f"Training samples ({train_generator.n}) don't match expected ({expected_train})")

if validation_generator.n == expected_val:
    print(f"Validation samples match expected: {expected_val}")
else:
    print(f"Validation samples ({validation_generator.n}) don't match expected ({expected_val})")

print("\n✓ STEP 1 COMPLETE: Data processing ready!")
print("="*60)

# ============================================================================
# STEP 2: NEURAL NETWORK ARCHITECTURE DESIGN
# ============================================================================

print("\n" + "="*60)
print("STEP 2: NEURAL NETWORK ARCHITECTURE DESIGN")
print("="*60)

# Build the CNN model
model = Sequential(name='Aircraft_Defect_Classifier')

print("\nBuilding Convolutional Neural Network...")

# First Convolutional Block
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', 
                 input_shape=IMG_SHAPE, name='conv1'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool1'))
print("✓ Block 1: 32 filters, 3x3 kernel")

# Second Convolutional Block
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='conv2'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool2'))
print("✓ Block 2: 64 filters, 3x3 kernel")

# Third Convolutional Block
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', name='conv3'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool3'))
print("✓ Block 3: 128 filters, 3x3 kernel")

# Fourth Convolutional Block
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', name='conv4'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool4'))
print("✓ Block 4: 128 filters, 3x3 kernel")

# Fifth Convolutional Block (NEW - ADDED FOR IMPROVEMENT)
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', name='conv5'))
model.add(MaxPooling2D(pool_size=(2, 2), name='pool5'))
print("✓ Block 5: 256 filters, 3x3 kernel")

# Flatten layer - convert 2D feature maps to 1D
model.add(Flatten(name='flatten'))

# Fully Connected (Dense) Layers
model.add(Dense(256, activation='relu', name='dense1'))
model.add(Dropout(0.5, name='dropout1'))  # Dropout to prevent overfitting
print("✓ Dense layer 1: 256 neurons with dropout (0.5)")

model.add(Dense(128, activation='relu', name='dense2'))
model.add(Dropout(0.5, name='dropout2'))
print("✓ Dense layer 2: 128 neurons with dropout (0.5)")

# Output layer - 3 neurons for 3 classes with softmax
model.add(Dense(NUM_CLASSES, activation='softmax', name='output'))
print("✓ Output layer: 3 neurons with softmax activation")

print("\n" + "="*60)
print("MODEL ARCHITECTURE")
print("="*60)
model.summary()
print("="*60)

print("\n✓ STEP 2 COMPLETE: Neural network architecture designed!")
print("="*60)

# ============================================================================
# STEP 3: HYPERPARAMETER ANALYSIS & MODEL COMPILATION
# ============================================================================

print("\n" + "="*60)
print("STEP 3: HYPERPARAMETER ANALYSIS & MODEL COMPILATION")
print("="*60)

# Compile the model with loss function and optimizer
print("\nCompiling model...")
print("  - Loss function: categorical_crossentropy")
print("  - Optimizer: adam")
print("  - Metrics: accuracy")

# Use custom optimizer with optimized learning rate
optimizer = Adam(learning_rate=0.0005)  # Lower learning rate

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,  # Use custom optimizer instead of 'adam'
    metrics=['accuracy']
)

print("✓ Model compiled successfully")

# Display hyperparameters
print("\n" + "="*60)
print("HYPERPARAMETERS SUMMARY")
print("="*60)
print("Convolutional Layers:")
print("  - Activation: ReLU")
print("  - Filters: 32 → 64 → 128 → 128 → 256")  # Updated
print("  - Kernel size: 3x3")
print("  - Pooling: MaxPooling 2x2")
print("\nDense Layers:")
print("  - Activation: ReLU (hidden layers)")
print("  - Activation: Softmax (output layer)")
print("  - Neurons: 256 → 128 → 3")
print("  - Dropout rate: 0.5")
print("\nTraining Parameters:")
print(f"  - Loss function: categorical_crossentropy")
print(f"  - Optimizer: adam (learning_rate=0.0005)")  # Updated
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Early stopping: patience=8")
print(f"  - Image size: {IMG_HEIGHT}x{IMG_WIDTH}x{IMG_CHANNELS}")
print("="*60)

print("\n✓ STEP 3 COMPLETE: Model compiled with hyperparameters!")
print("="*60)

# ============================================================================
# STEP 4: MODEL TRAINING & EVALUATION
# ============================================================================

print("\n" + "="*60)
print("STEP 4: MODEL TRAINING & EVALUATION")
print("="*60)

# Set training parameters
EPOCHS = 30  # Increased, but early stopping will stop sooner if needed

# Add callbacks for smarter training
print("\nSetting up training callbacks...")
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1
    )
]
print("✓ Early stopping: patience=8 epochs")
print("✓ Learning rate reduction: factor=0.5, patience=4")

print(f"\nTraining configuration:")
print(f"  - Epochs: {EPOCHS}")
print(f"  - Steps per epoch: {train_generator.n // BATCH_SIZE}")
print(f"  - Validation steps: {validation_generator.n // BATCH_SIZE}")

print("\n" + "="*60)
print("Starting model training...")
print("This will take a while. Please wait...")
print("="*60 + "\n")

# Start timing
start_time = time.time()
print(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,  
    verbose=1
)

# End timing
end_time = time.time()
total_time = end_time - start_time

# Convert to hours, minutes, seconds
hours = int(total_time // 3600)
minutes = int((total_time % 3600) // 60)
seconds = int(total_time % 60)

print("\n" + "="*60)
print("✓ Training complete!")
print("="*60)
print(f"Training ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total training time: {hours}h {minutes}m {seconds}s ({total_time:.2f} seconds)")
print(f"Average time per epoch: {total_time/EPOCHS:.2f} seconds")
print("="*60)

# Save the trained model
model_save_path = 'models/aircraft_defect_model.keras'
os.makedirs('models', exist_ok=True)
model.save(model_save_path)
print(f"✓ Model saved to: {model_save_path}")

# ============================================================================
# GENERATE PERFORMANCE PLOTS
# ============================================================================

print("\n" + "="*60)
print("GENERATING PERFORMANCE PLOTS...")
print("="*60)

# Create output directory
os.makedirs('outputs', exist_ok=True)

# Plot training & validation accuracy
plt.figure(figsize=(14, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = 'outputs/model_performance.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Performance plot saved to: {plot_path}")
plt.close()

# Print final metrics
print("\n" + "="*60)
print("FINAL TRAINING METRICS")
print("="*60)
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print(f"Final Training Accuracy:   {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
print(f"Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
print(f"Final Training Loss:       {final_train_loss:.4f}")
print(f"Final Validation Loss:     {final_val_loss:.4f}")
print("="*60)

# Check for overfitting
if final_train_acc - final_val_acc > 0.15:
    print("\n WARNING: Possible overfitting detected!")
    print("   Training accuracy significantly higher than validation accuracy.")
    print("   Consider: increasing dropout, adding more data augmentation,")
    print("   or reducing model complexity.")
elif final_val_acc > final_train_acc:
    print("\n✓ Good generalization: Validation accuracy >= Training accuracy")
else:
    print("\n✓ Model appears to be learning well")

print("\n✓ STEP 4 COMPLETE: Model trained and evaluated!")
print("="*60)

print("\n" + "="*60)
print("TRAINING PIPELINE COMPLETE!")
print("="*60)