# import streamlit as st
# import requests
# import io
# from PIL import Image

# # Centralized location for our FastAPI local engine service
# BACKEND_URL = "http://127.0.0.1:8000"

# st.set_page_config(layout="wide", page_title="Custom Teachable Machine")
# st.title("🧠 Decoupled Teachable Machine Clone (TensorFlow Engine)")

# # Initialize state trackers so memory isn't erased on re-renders
# if "classes" not in st.session_state:
#     st.session_state.classes = ["Class A", "Class B"]
# if "model_trained" not in st.session_state:
#     st.session_state.model_trained = False

# # Layout Structure splits the dashboard into two vertical workspaces
# col_left, col_right = st.columns([2, 1])

# with col_left:
#     st.header("1. Dataset Ingestion Workspace")
    
#     # Render interactive input panels for each class dynamically
#     for idx, class_item in enumerate(st.session_state.classes):
#         with st.container(border=True):
#             # Dynamic naming field
#             new_name = st.text_input(f"Define Label Category #{idx+1}", value=class_item, key=f"input_{idx}")
#             st.session_state.classes[idx] = new_name
            
#             # Support both batch file dropping or direct hardware camera captures
#             upload_type = st.radio("Input Vector Source", ["Webcam Frame", "File Upload"], key=f"type_{idx}", horizontal=True)
            
#             uploaded_files = []
#             if upload_type == "File Upload":
#                 files = st.file_uploader("Select image files...", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=f"file_{idx}")
#                 if files:
#                     uploaded_files = files
#             else:
#                 cam_frame = st.camera_input("Capture instant frame snapshot", key=f"cam_{idx}")
#                 if cam_frame:
#                     uploaded_files = [cam_frame]
            
#             # Send sample images to the backend when clicked
#             if st.button(f"Upload Data Samples to Category: {new_name}", key=f"btn_{idx}"):
#                 if not uploaded_files:
#                     st.warning("Please capture or upload data frames first.")
#                 else:
#                     files_payload = [("files", (f.name if hasattr(f, 'name') else "frame.jpg", f.read(), "image/jpeg")) for f in uploaded_files]
#                     data_payload = {"class_name": new_name}
                    
#                     with st.spinner("Streaming images to FastAPI backend server..."):
#                         response = requests.post(f"{BACKEND_URL}/upload-sample", data=data_payload, files=files_payload)
#                         if response.status_code == 200:
#                             st.success(response.json()["message"])
#                         else:
#                             st.error(f"Upload failed: {response.text}")

#     if st.button("➕ Add Alternative Label Category"):
#         st.session_state.classes.append(f"Class {chr(65 + len(st.session_state.classes))}")
#         st.rerun()

# with col_right:
#     st.header("2. AI Operations Control")
    
#     # Trigger the model training loop
#     if st.button("⚡ Execute Fast Model Training", use_container_width=True, type="primary"):
#         with st.spinner("Extracting TensorFlow features & optimizing classification plane..."):
#             res = requests.post(f"{BACKEND_URL}/train")
#             if res.status_code == 200:
#                 st.session_state.model_trained = True
#                 st.success(res.json()["message"])
#             else:
#                 # Safely catches single-class issues or empty folders without crashing Streamlit
#                 st.error(f"Training Rejection: {res.json().get('detail', 'Unknown error')}")

#     st.markdown("---")
    
#     # Real-Time Prediction UI (hidden until model is trained)
#     if st.session_state.model_trained:
#         st.header("3. Live Classification Test")
#         test_frame = st.camera_input("Stream active verification frame", key="inference_test_camera")
        
#         if test_frame:
#             inference_file = [("file", ("test.jpg", test_frame.read(), "image/jpeg"))]
#             inf_res = requests.post(f"{BACKEND_URL}/predict", files=inference_file)
            
#             if inf_res.status_code == 200:
#                 data = inf_res.json()
#                 st.subheader(f"Prediction: **{data['prediction']}**")
                
#                 # Render real-time bar graphs tracking precision distributions
#                 for label, confidence in data["probabilities"].items():
#                     st.write(f"{label} ({confidence*100:.1f}%)")
#                     st.progress(int(confidence * 100))
#             else:
#                 st.error("Inference processing error via core server pipeline.")
#     else:
#         st.info("Awaiting execution cycle. Add dataset classes and train to initialize real-time predictions.")
import streamlit as st
import os
import uuid
import io
import joblib
from PIL import Image

# Direct internal module imports instead of API requests
from app import config
from app.engine import TeachableEngine

st.set_page_config(layout="wide", page_title="Teachable Machine Cloud")
st.title("🧠 Custom Teachable Machine (Cloud Deployment)")

# Initialize Engine directly inside Streamlit
@st.cache_resource
def load_engine():
    return TeachableEngine()

engine = load_engine()

if "classes" not in st.session_state:
    st.session_state.classes = ["Class A", "Class B"]
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

col_left, col_right = st.columns([2, 1])

with col_left:
    st.header("1. Dataset Ingestion")
    
    for idx, class_item in enumerate(st.session_state.classes):
        with st.container(border=True):
            new_name = st.text_input(f"Category #{idx+1}", value=class_item, key=f"input_{idx}")
            st.session_state.classes[idx] = new_name
            
            upload_type = st.radio("Source", ["Webcam", "File Upload"], key=f"type_{idx}", horizontal=True)
            
            uploaded_files = []
            if upload_type == "File Upload":
                files = st.file_uploader("Drop images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key=f"file_{idx}")
                if files:
                    uploaded_files = files
            else:
                cam_frame = st.camera_input("Take Snapshot", key=f"cam_{idx}")
                if cam_frame:
                    uploaded_files = [cam_frame]
            
            if st.button(f"Save Samples to {new_name}", key=f"btn_{idx}"):
                if not uploaded_files:
                    st.warning("No images captured.")
                else:
                    # Replace backend API call with simple local file saving logic
                    class_folder = os.path.join(config.DATA_DIR, new_name)
                    os.makedirs(class_folder, exist_ok=True)
                    
                    saved_count = 0
                    for f in uploaded_files:
                        try:
                            img = Image.open(f)
                            unique_filename = f"{uuid.uuid4().hex}.jpg"
                            img.convert("RGB").save(os.path.join(class_folder, unique_filename), "JPEG")
                            saved_count += 1
                        except Exception:
                            continue
                    st.success(f"Saved {saved_count} frames to {new_name} locally on cloud runtime.")

    if st.button("➕ Add Category"):
        st.session_state.classes.append(f"Class {chr(65 + len(st.session_state.classes))}")
        st.rerun()

with col_right:
    st.header("2. AI Operations")
    
    if st.button("⚡ Train Model", use_container_width=True, type="primary"):
        with st.spinner("TensorFlow extracting features..."):
            try:
                # Call local method directly
                msg = engine.train_classifier()
                st.session_state.model_trained = True
                st.success(msg)
            except ValueError as e:
                st.error(str(e))

    st.markdown("---")
    
    if st.session_state.model_trained:
        st.header("3. Live Testing")
        test_frame = st.camera_input("Verification Stream", key="test_cam")
        
        if test_frame:
            try:
                # Local inference shortcut mimicking the predict API route
                classifier = joblib.load(config.MODEL_PATH)
                temp_path = os.path.join(config.MODEL_DIR, "temp_inference.jpg")
                
                img = Image.open(test_frame)
                img.convert("RGB").save(temp_path, "JPEG")
                
                features = engine.extract_features(temp_path)
                probabilities = classifier.predict_proba([features])[0]
                classes = classifier.classes_
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                
                prediction = classifier.predict([features])[0]
                st.subheader(f"Result: **{prediction}**")
                
                for i, label in enumerate(classes):
                    st.write(f"{label} ({probabilities[i]*100:.1f}%)")
                    st.progress(int(probabilities[i] * 100))
            except Exception as e:
                st.error(f"Inference failure: {e}")
