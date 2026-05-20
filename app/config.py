import os

# Base project directory (two levels up from this app package)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Where uploaded/training images are stored
DATA_DIR = os.path.join(BASE_DIR, "data")

# Models directory and default model path
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "classifier.joblib")

# Image size expected by the MobileNetV3 backbone
IMAGE_SIZE = (224, 224)

# Optional: limit TensorFlow GPU visibility if needed
TF_ALLOW_GPU = os.environ.get("TF_ALLOW_GPU", "1")
