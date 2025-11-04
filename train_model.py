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

