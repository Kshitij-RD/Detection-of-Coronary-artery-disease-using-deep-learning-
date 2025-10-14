import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import shutil

# Set page configuration
st.set_page_config(
    page_title="Coronary Artery Disease Detection",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #ff4b4b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
    }
    .normal {
        background-color: #d4edda;
        border: 2px solid #c3e6cb;
        color: #155724;
    }
    .sick {
        background-color: #f8d7da;
        border: 2px solid #f5c6cb;
        color: #721c24;
    }
    .confidence-bar {
        height: 20px;
        background-color: #e9ecef;
        border-radius: 10px;
        margin: 10px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    .model-info {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

class CoronaryArteryApp:
    def __init__(self):
        self.model = None
        self.class_names = ['Normal', 'Sick']
        self.model_loaded = False
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        # Try different model paths
        model_paths = [
            'models/trained_models/coronary_simple_cnn_best.h5',
            'models/trained_models/coronary_simple_cnn_final.h5',
            'models/trained_models/coronary_cnn_best.h5',
            'models/trained_models/coronary_cnn_final.h5'
        ]
        
        loaded_model = None
        loaded_path = None
        
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    loaded_model = tf.keras.models.load_model(model_path)
                    loaded_path = model_path
                    st.sidebar.success(f"✅ Model loaded successfully from: {os.path.basename(model_path)}")
                    break
                except Exception as e:
                    st.sidebar.warning(f"⚠️ Could not load {model_path}: {e}")
                    continue
        
        if loaded_model is not None:
            self.model = loaded_model
            self.model_loaded = True
            st.sidebar.info(f"**Model Info:**")
            st.sidebar.text(f"Input Shape: {self.model.input_shape[1:3]}")
            st.sidebar.text(f"Classes: {len(self.class_names)}")
            st.sidebar.text(f"Model Type: {loaded_path}")
        else:
            st.sidebar.error("❌ No trained model found. Please train the model first.")
            st.sidebar.info("Run: `python train_model.py` to train a model")
    
    def preprocess_image(self, image, img_size=(224, 224)):
        """Preprocess image for model prediction"""
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Ensure image has 3 channels
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Resize image
            image = cv2.resize(image, img_size)
            
            # Normalize pixel values
            image = image.astype('float32') / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            st.error(f"Error preprocessing image: {e}")
            return None
    
    def predict_image(self, image):
        """Predict class for a single image"""
        if not self.model_loaded:
            raise ValueError("Model not loaded. Please ensure the model file exists.")
        
        # Preprocess image
        processed_img = self.preprocess_image(image)
        
        if processed_img is None:
            raise ValueError("Could not preprocess the image")
        
        # Make prediction
        predictions = self.model.predict(processed_img, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        return {
            'class': self.class_names[predicted_class_idx],
            'confidence': float(confidence),
            'all_predictions': predictions[0].tolist(),
            'class_index': predicted_class_idx
        }
    
    def display_prediction_results(self, prediction, image, original_filename=""):
        """Display prediction results with visualization"""
        confidence = prediction['confidence']
        predicted_class = prediction['class']
        all_predictions = prediction['all_predictions']
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Uploaded Image")
            if isinstance(image, Image.Image):
                st.image(image, caption=f"Image: {original_filename}", use_column_width=True)
            else:
                # Convert numpy array to PIL Image for display
                display_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                st.image(display_image, caption=f"Image: {original_filename}", use_column_width=True)
        
        with col2:
            st.subheader("🔍 Prediction Results")
            
            # Prediction box
            if predicted_class == 'Normal':
                box_class = "normal"
                emoji = "✅"
                message = "No signs of coronary artery disease detected."
                confidence_color = "#28a745"
            else:
                box_class = "sick"
                emoji = "⚠️"
                message = "Potential signs of coronary artery disease detected. Please consult a healthcare professional."
                confidence_color = "#dc3545"
            
            st.markdown(f"""
            <div class="prediction-box {box_class}">
                <h2>{emoji} {predicted_class}</h2>
                <h3>Confidence: {confidence:.2%}</h3>
                <p>{message}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence visualization
            st.subheader("📈 Confidence Scores")
            
            for i, (class_name, pred_score) in enumerate(zip(self.class_names, all_predictions)):
                color = "#28a745" if class_name == "Normal" else "#dc3545"
                st.write(f"**{class_name}:** {pred_score:.2%}")
                
                # Confidence bar
                st.markdown(f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {pred_score*100}%; background-color: {color};"></div>
                </div>
                """, unsafe_allow_html=True)
    
    def plot_confidence_chart(self, prediction):
        """Plot confidence scores as a bar chart"""
        fig, ax = plt.subplots(figsize=(8, 4))
        
        classes = self.class_names
        confidences = prediction['all_predictions']
        
        colors = ['green' if cls == 'Normal' else 'red' for cls in classes]
        bars = ax.bar(classes, confidences, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_ylabel('Confidence Score', fontsize=12)
        ax.set_title('Prediction Confidence by Class', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, confidence in zip(bars, confidences):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{confidence:.2%}', ha='center', va='bottom', fontweight='bold')
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        st.pyplot(fig)
    
    def run(self):
        """Run the Streamlit application"""
        # Header
        st.markdown('<h1 class="main-header">❤️ Coronary Artery Disease Detection</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        st.sidebar.title("🔧 Navigation")
        app_mode = st.sidebar.selectbox("Choose Mode", 
                                       ["Single Image Prediction", "Batch Prediction", "Model Information", "About"])
        
        if app_mode == "Single Image Prediction":
            self.single_image_prediction()
        elif app_mode == "Batch Prediction":
            self.batch_prediction()
        elif app_mode == "Model Information":
            self.model_information()
        else:
            self.about_page()
    
    def single_image_prediction(self):
        """Single image prediction interface"""
        st.header("🖼️ Single Image Prediction")
        
        if not self.model_loaded:
            st.error("Please train a model first or ensure model files exist in 'models/trained_models/' directory.")
            return
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 Upload Image")
            uploaded_file = st.file_uploader("Choose a medical scan image", 
                                           type=['jpg', 'jpeg', 'png'],
                                           help="Upload a coronary artery scan image for analysis")
            
            if uploaded_file is not None:
                try:
                    # Display uploaded image
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                    
                    # Add prediction button
                    if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                        with st.spinner("🔄 Analyzing image..."):
                            try:
                                # Make prediction
                                prediction = self.predict_image(image)
                                
                                # Display results
                                self.display_prediction_results(prediction, image, uploaded_file.name)
                                
                                # Plot confidence chart
                                st.subheader("📊 Confidence Visualization")
                                self.plot_confidence_chart(prediction)
                                
                            except Exception as e:
                                st.error(f"❌ Error during prediction: {e}")
                
                except Exception as e:
                    st.error(f"❌ Error processing uploaded file: {e}")
        
        with col2:
            if uploaded_file is None:
                st.info("""
                👆 **Please upload an image to get started**
                
                **Supported formats:** JPG, JPEG, PNG
                **Expected content:** Coronary artery scan images
                
                Once you upload an image, click the 'Analyze Image' button to get the prediction.
                """)
    
    def batch_prediction(self):
        """Batch prediction interface"""
        st.header("📚 Batch Prediction")
        
        if not self.model_loaded:
            st.error("Please train a model first or ensure model files exist in 'models/trained_models/' directory.")
            return
        
        st.subheader("📤 Upload Multiple Images")
        uploaded_files = st.file_uploader("Choose multiple medical scan images", 
                                        type=['jpg', 'jpeg', 'png'], 
                                        accept_multiple_files=True,
                                        help="Upload multiple coronary artery scan images for batch analysis")
        
        if uploaded_files:
            if st.button("🔍 Analyze All Images", type="primary"):
                if self.model_loaded:
                    with st.spinner(f"🔄 Analyzing {len(uploaded_files)} images..."):
                        results = []
                        valid_images = []
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, uploaded_file in enumerate(uploaded_files):
                            status_text.text(f"Processing image {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                            
                            try:
                                image = Image.open(uploaded_file)
                                prediction = self.predict_image(image)
                                prediction['filename'] = uploaded_file.name
                                results.append(prediction)
                                valid_images.append((uploaded_file.name, image))
                                
                            except Exception as e:
                                st.warning(f"⚠️ Could not process {uploaded_file.name}: {e}")
                            
                            progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        status_text.text("✅ Analysis complete!")
                        
                        # Display batch results
                        self.display_batch_results(results, valid_images)
                else:
                    st.error("Model not loaded. Please ensure the model file exists.")
    
    def display_batch_results(self, results, valid_images):
        """Display batch prediction results"""
        st.subheader("📈 Batch Analysis Results")
        
        if not results:
            st.warning("No valid results to display.")
            return
        
        # Summary statistics
        normal_count = sum(1 for r in results if r['class'] == 'Normal')
        sick_count = len(results) - normal_count
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Images", len(results))
        with col2:
            st.metric("✅ Normal", normal_count, delta=f"{(normal_count/len(results)*100):.1f}%")
        with col3:
            st.metric("⚠️ Sick", sick_count, delta=f"{(sick_count/len(results)*100):.1f}%")
        with col4:
            avg_confidence = np.mean([r['confidence'] for r in results])
            st.metric("🎯 Avg Confidence", f"{avg_confidence:.2%}")
        
        # Detailed results
        st.subheader("📋 Detailed Results")
        
        for i, (result, (filename, image)) in enumerate(zip(results, valid_images)):
            with st.expander(f"Image {i+1}: {filename} - **{result['class']}** ({result['confidence']:.2%})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption=filename, use_column_width=True)
                
                with col2:
                    predicted_class = result['class']
                    confidence = result['confidence']
                    
                    if predicted_class == 'Normal':
                        st.success(f"**Prediction:** {predicted_class}")
                    else:
                        st.error(f"**Prediction:** {predicted_class}")
                    
                    st.write(f"**Confidence:** {confidence:.2%}")
                    
                    # Confidence breakdown
                    st.write("**Confidence Breakdown:**")
                    for j, class_name in enumerate(self.class_names):
                        conf = result['all_predictions'][j]
                        color = "green" if class_name == "Normal" else "red"
                        st.markdown(f"- <span style='color:{color}'>{class_name}: {conf:.2%}</span>", 
                                   unsafe_allow_html=True)
    
    def model_information(self):
        """Display model information"""
        st.header("🤖 Model Information")
        
        if not self.model_loaded:
            st.error("No model is currently loaded.")
            st.info("""
            To use the application:
            1. Train a model using: `python train_model.py`
            2. Ensure model files are saved in `models/trained_models/`
            3. Restart the application
            """)
            return
        
        st.markdown("""
        <div class="model-info">
        <h3>📊 Model Details</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Architecture")
            if self.model:
                st.text(f"Input Shape: {self.model.input_shape}")
                st.text(f"Output Shape: {self.model.output_shape}")
                st.text(f"Number of Layers: {len(self.model.layers)}")
                st.text(f"Trainable Parameters: {self.model.count_params():,}")
        
        with col2:
            st.subheader("Classification Info")
            st.text(f"Number of Classes: {len(self.class_names)}")
            st.text(f"Classes: {', '.join(self.class_names)}")
        
        # Show model summary
        st.subheader("Model Summary")
        if self.model:
            # Create a string buffer to capture model summary
            string_list = []
            self.model.summary(print_fn=lambda x: string_list.append(x))
            summary_string = "\n".join(string_list)
            st.text_area("Model Architecture", summary_string, height=300)
    
    def about_page(self):
        """About page content"""
        st.header("ℹ️ About This Application")
        
        st.markdown("""
        ## Coronary Artery Disease Detection System
        
        This application uses a Convolutional Neural Network (CNN) to detect signs of 
        coronary artery disease from medical scan images.
        
        ### 🎯 How it works:
        1. **Image Upload**: Users can upload medical scan images (JPG, JPEG, PNG)
        2. **AI Analysis**: The CNN model analyzes the image patterns
        3. **Results**: Returns prediction with confidence scores
        
        ### 📊 Classes:
        - **Normal**: No signs of coronary artery disease
        - **Sick**: Potential signs of coronary artery disease detected
        
        ### ⚠️ Important Medical Disclaimer:
        > **This tool is for educational and research purposes only.**
        > - Always consult healthcare professionals for medical diagnosis
        > - The model was trained on a limited dataset
        > - Results should not be used as a substitute for professional medical advice
        > - False positives and false negatives are possible
        
        ### 🔧 Technical Details:
        - **Framework**: TensorFlow/Keras
        - **Model**: Convolutional Neural Network (CNN)
        - **Input Size**: 224x224 pixels
        - **Preprocessing**: Image normalization and resizing
        - **Supported Formats**: JPG, JPEG, PNG
        
        ### 📁 Project Structure:
        ```
        coronary-artery-disease-detection/
        ├── data/               # Dataset directory
        ├── models/             # Trained models
        ├── src/               # Source code
        ├── app.py            # This Streamlit app
        └── train_model.py    # Training script
        ```
        
        ### 🚀 Getting Started:
        1. Train the model: `python train_model.py`
        2. Run the app: `streamlit run app.py`
        3. Upload images for analysis
        """)
        
        # Model status
        if self.model_loaded:
            st.success("✅ Model is loaded and ready for predictions")
        else:
            st.warning("⚠️ Model not loaded. Please train the model first.")

def main():
    # Initialize and run the application
    app = CoronaryArteryApp()
    app.run()

if __name__ == "__main__":
    main()