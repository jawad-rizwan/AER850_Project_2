# =============================================================
# AER 850 Project 2: Aircraft Defect Classification System (Step 5)
# Name: Jawad Rizwan
# Student Number: 501124033
# Due Date: November 5th, 2025
# -------------------------------------------------------------
# This script loads the trained CNN model and tests it on three specific defect images,
# generating predictions with confidence scores and visualization outputs.
# 
# Inputs:
#     - Trained model: models/aircraft_defect_model.keras
#     - Test images: Data/test/[defect-type]/test_[defect].jpg
# 
# Outputs:
#     - Prediction visualizations: outputs/test_result_[defect-type].png
#     - Test accuracy metrics printed to console
# =============================================================

# Import os and suppress TensorFlow warnings 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# ============================================================================
# STEP 5: MODEL TESTING
# ============================================================================

print("="*60)
print("STEP 5: MODEL TESTING")
print("="*60)

# Load the trained model
MODEL_PATH = 'models/aircraft_defect_model.keras'  # or .h5 
print(f"\nLoading trained model from: {MODEL_PATH}")

try:
    model = load_model(MODEL_PATH)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure you've trained the model first (run train_model.py)")
    exit(1)

# Define class labels (must match training order)
class_labels = ['crack', 'missing-head', 'paint-off']

# Define test image paths (as specified in project)
test_images = {
    'crack': 'Data/test/crack/test_crack.jpg',
    'missing-head': 'Data/test/missing-head/test_missinghead.jpg',
    'paint-off': 'Data/test/paint-off/test_paintoff.jpg'
}

# Image parameters (must match training)
IMG_HEIGHT = 500
IMG_WIDTH = 500

print("\n" + "="*60)
print("TESTING ON NEW IMAGES")
print("="*60)

# Create output directory
os.makedirs('outputs', exist_ok=True)

# Process each test image
for true_label, img_path in test_images.items():
    print(f"\nProcessing: {img_path}")
    
    # Check if image exists
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue
    
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_probs = predictions[0]
    
    # Get predicted class
    predicted_class_idx = np.argmax(predicted_probs)
    predicted_class = class_labels[predicted_class_idx]
    confidence = predicted_probs[predicted_class_idx] * 100
    
    # Print results
    print(f"  True Label: {true_label}")
    print(f"  Predicted: {predicted_class} ({confidence:.1f}% confidence)")
    print(f"  All probabilities:")
    for i, label in enumerate(class_labels):
        print(f"    - {label}: {predicted_probs[i]*100:.1f}%")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display the image
    display_img = image.load_img(img_path)
    ax.imshow(display_img)
    ax.axis('off')
    
    # Add prediction text on image
    text_str = f"True Label: {true_label}\n"
    text_str += f"Predicted: {predicted_class}\n\n"
    text_str += f"Crack: {predicted_probs[0]*100:.1f}%\n"
    text_str += f"Missing Head: {predicted_probs[1]*100:.1f}%\n"
    text_str += f"Paint-off: {predicted_probs[2]*100:.1f}%"
    
    # Position text on image with background
    ax.text(0.02, 0.98, text_str,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            color='green' if predicted_class == true_label else 'red',
            weight='bold')
    
    # Add title
    if predicted_class == true_label:
        title_color = 'green'
        title = f"✓ CORRECT PREDICTION"
    else:
        title_color = 'red'
        title = f"✗ INCORRECT PREDICTION"
    
    plt.title(title, fontsize=16, fontweight='bold', color=title_color, pad=20)
    
    # Save the figure
    output_path = f'outputs/test_result_{true_label}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Result saved to: {output_path}")
    plt.close()

print("\n" + "="*60)
print("✓ STEP 5 COMPLETE: All test images processed!")
print("="*60)

# Calculate test accuracy
print("\n" + "="*60)
print("TEST SET SUMMARY")
print("="*60)

correct = 0
total = len(test_images)

for true_label, img_path in test_images.items():
    if not os.path.exists(img_path):
        continue
    
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_labels[predicted_class_idx]
    
    if predicted_class == true_label:
        correct += 1

accuracy = (correct / total) * 100
print(f"Test Accuracy: {correct}/{total} correct ({accuracy:.1f}%)")
print("="*60)

print("\nVisualized results saved in the 'outputs' directory.")