# AER850 Project 2: Aircraft Defect Classification System

**Author:** Jawad Rizwan  
**Student Number:** 501124033  
**Due Date:** November 5th, 2025

## Overview

This project develops a Deep Convolutional Neural Network (DCNN) to automatically classify aircraft surface defects into three categories: crack, missing-head, and paint-off. The workflow includes data processing, CNN architecture design, hyperparameter tuning, model training, and testing on new images.

## Features

* Loads and preprocesses 1,942 training images and 431 validation images
* Implements conservative data augmentation for realistic defect preservation
* Builds a 5-layer CNN with progressive filter architecture (32→64→128→128→256)
* Compiles model with optimized Adam optimizer (learning_rate=0.0005)
* Implements early stopping and learning rate reduction callbacks
* Trains model with performance tracking and visualization
* Generates accuracy and loss plots for training/validation
* Tests model on 3 specific defect images with confidence scores
* Creates visual predictions overlaid on test images

## Libraries Used

* **tensorflow/keras:** Model building, training, and evaluation
* **numpy:** Numerical operations and array handling
* **pandas:** Data manipulation
* **matplotlib:** Performance visualization and result plotting
* **scipy:** Required for image augmentation transformations

## How to Run

1. Ensure dataset is organized in `Data/train/`, `Data/valid/`, and `Data/test/` directories
2. Install required libraries: `pip install tensorflow numpy pandas matplotlib scipy`
3. Train the model: `python train_model.py` (~2.5 hours on CPU)
4. Test the model: `python test_model.py`
5. Review performance plots in `outputs/` directory

## File Structure

* `train_model.py` - Data processing, model architecture, training, and evaluation (Steps 1-4)
* `test_model.py` - Model testing on new images (Step 5)
* `Data/train/` - Training images (1,942 images across 3 classes)
* `Data/valid/` - Validation images (431 images across 3 classes)
* `Data/test/` - Test images (539 images, 3 used for testing)
* `models/` - Saved trained models (excluded from repo due to size)
* `outputs/` - Performance plots and test result visualizations

## Notes

* Model file (`.keras`) is excluded from the repository due to large file size (>300 MB)
* All hyperparameters are tuned based on validation performance
* Training uses CPU and takes approximately 2.5 hours
* Conservative data augmentation strategy aligns with controlled inspection environments
* Results and visualizations are saved for easy interpretation