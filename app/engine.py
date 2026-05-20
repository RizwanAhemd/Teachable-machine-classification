import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.linear_model import LogisticRegression
import joblib
from app import config

class TeachableEngine:
    def __init__(self):
        # 1. Load pre-trained MobileNetV3 (without the final classification layer)
        # pooling='avg' automatically flattens the output into a single feature vector
        self.base_model = MobileNetV2(
            weights='imagenet', 
            include_top=False, 
            pooling='avg'
        )

    def extract_features(self, image_path: str):
        """Opens an image, preprocesses it for TensorFlow, and extracts features using CPU."""
        # Load image and force it to the required 224x224 shape
        img = load_img(image_path, target_size=config.IMAGE_SIZE)
        img_array = img_to_array(img)
        
        # Add a batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        
        # Scale pixels dynamically based on MobileNetV3's exact expectations
        preprocessed_img = preprocess_input(expanded_img_array)
        
        # Extract features using the base model
        features = self.base_model.predict(preprocessed_img, verbose=0)
        
        # Flatten into a clean 1D numpy array vector
        return features.flatten()

    def train_classifier(self):
        """Scans local folders, extracts features via TF, and fits a fast Logistic Regression model."""
        X = []
        y = []
        
        # Scan data directory for class folders
        classes = [c for c in os.listdir(config.DATA_DIR) if os.path.isdir(os.path.join(config.DATA_DIR, c))]
        
        if len(classes) < 2:
            raise ValueError("You must upload samples for at least 2 distinct classes before training.")
            
        for class_label in classes:
            class_path = os.path.join(config.DATA_DIR, class_label)
            images_list = [img for img in os.listdir(class_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_name in images_list:
                img_path = os.path.join(class_path, img_name)
                try:
                    features = self.extract_features(img_path)
                    X.append(features)
                    y.append(class_label)
                except Exception:
                    continue  # Skip broken or corrupt files gracefully
                    
        if len(X) == 0:
            raise ValueError("No valid training images found in your dataset directories.")

        # Train a Logistic Regression model on top of our extracted TF features
        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(X, y)
        
        # Save the trained model to disk
        joblib.dump(classifier, config.MODEL_PATH)
        return f"Successfully trained on {len(X)} samples across {len(classes)} classes using TensorFlow backbone."