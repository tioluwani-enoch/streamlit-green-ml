"""
GREEN-ML - Beautiful Streamlit App
Minimalistic camera interface with instant classification
"""

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="GREEN-ML",
    page_icon="🗑️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for beautiful design
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* White content box */
    .block-container {
        background: white;
        border-radius: 20px;
        padding: 3rem 2rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        max-width: 800px;
    }
    
    /* Title */
    h1 {
        font-size: 3rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 0.5rem;
        color: #000;
        letter-spacing: -1px;
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Instruction box */
    .instruction-box {
        background: #e3f2fd;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #1565c0;
        margin-bottom: 2rem;
    }
    
    .instruction-box p {
        color: #1565c0;
        margin: 0;
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* Results box */
    .result-box {
        background: #000;
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin-top: 2rem;
    }
    
    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .result-item {
        background: rgba(255,255,255,0.1);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .result-category {
        font-size: 1.3rem;
        font-weight: 700;
    }
    
    .result-confidence {
        font-size: 1.5rem;
        font-weight: 800;
    }
    
    /* Category cards */
    .category-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .category-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        color: white;
    }
    
    .category-card.compost {
        background: linear-gradient(135deg, #66bb6a 0%, #43a047 100%);
    }
    
    .category-card.recycle {
        background: linear-gradient(135deg, #42a5f5 0%, #1976d2 100%);
    }
    
    .category-card.landfill {
        background: linear-gradient(135deg, #ffa726 0%, #f57c00 100%);
    }
    
    .category-card h3 {
        margin: 0 0 0.5rem 0;
        font-size: 1.5rem;
    }
    
    .category-card p {
        margin: 0;
        font-size: 0.9rem;
        opacity: 0.95;
    }
    
    /* Camera button */
    .stButton button {
        background: #4CAF50 !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 1rem 2rem !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        width: 100%;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        background: #45a049 !important;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(76, 175, 80, 0.4);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = tf.keras.models.load_model('waste_sorting_model_compatible.h5', compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def classify_image(image, model):
    """Classify the waste image"""
    try:
        # Resize and preprocess
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get results
        classes = ['Compost', 'Recycle', 'Landfill']
        emojis = ['🍌', '♻️', '🗑️']
        
        results = []
        for i, (cls, emoji, pred) in enumerate(zip(classes, emojis, predictions)):
            results.append({
                'emoji': emoji,
                'class': cls,
                'confidence': pred
            })
        
        # Sort by confidence
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return results
        
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return None

# Main app
def main():
    # Title
    st.markdown("# 🗑️ GREEN-ML")
    st.markdown('<p class="subtitle">Point, shoot, classify. That\'s it.</p>', unsafe_allow_html=True)
    
    # Instructions
    st.markdown("""
    <div class="instruction-box">
        <p>📸 Take a photo or upload an image of waste → Get instant classification</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("⚠️ Model not loaded. Please check model file.")
        return
    
    # Camera input
    camera_image = st.camera_input("📷 Take a picture", label_visibility="collapsed")
    
    # Also allow file upload
    st.markdown("**Or upload an image:**")
    uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    
    # Use whichever is available
    image_source = camera_image if camera_image else uploaded_file
    
    if image_source:
        # Open image
        image = Image.open(image_source)
        
        # Display image
        st.image(image, use_container_width=True)
        
        # Show loading
        with st.spinner('🔍 Analyzing...'):
            results = classify_image(image, model)
        
        if results:
            # Display results
            st.markdown(f"""
            <div class="result-box">
                <div class="result-title">🎯 CLASSIFICATION RESULTS</div>
            """, unsafe_allow_html=True)
            
            for result in results:
                confidence_pct = result['confidence'] * 100
                st.markdown(f"""
                <div class="result-item">
                    <span class="result-category">{result['emoji']} {result['class'].upper()}</span>
                    <span class="result-confidence">{confidence_pct:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Show winner
            winner = results[0]
            if winner['confidence'] > 0.7:
                st.success(f"**Verdict:** This item belongs in **{winner['emoji']} {winner['class'].upper()}**")
            else:
                st.warning("**Note:** Low confidence. Item might be ambiguous or contaminated.")
    
    # Category info
    st.markdown("---")
    st.markdown("""
    <div class="category-grid">
        <div class="category-card compost">
            <h3>🍌 Compost</h3>
            <p>Food scraps, yard waste, organics</p>
        </div>
        <div class="category-card recycle">
            <h3>♻️ Recycle</h3>
            <p>Paper, plastic, metal, glass</p>
        </div>
        <div class="category-card landfill">
            <h3>🗑️ Landfill</h3>
            <p>Mixed waste, non-recyclables</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tips
    with st.expander("💡 Tips for Best Results"):
        st.markdown("""
        - **Use good lighting** - Natural light works best
        - **One item at a time** - Don't mix multiple items
        - **Center the item** - Fill most of the frame
        - **Avoid blur** - Hold camera steady
        - **Clean items** - Dirty items are harder to classify
        """)
    
    # Technical details
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        **Model Architecture:** MobileNetV2 with Transfer Learning  
        **Framework:** TensorFlow 2.15 / Keras  
        **Training Data:** 2000+ labeled waste images  
        **Validation Accuracy:** ~85%  
        **Input Size:** 224x224 RGB  
        **Classes:** 3 (Compost, Recycle, Landfill)
        """)

if __name__ == "__main__":
    main()