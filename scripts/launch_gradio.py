import os
import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tensorflow.keras.models import Model
from matplotlib import gridspec
import matplotlib as mpl
import tempfile
import io

# Configuration
IM_SIZE = 224
NUM_CLASSES = 2
CLASS_NAMES = ['no wildfire', 'wildfire']

def load_model_and_weights(model_path):
    """Load the pre-trained model"""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def prepare_cam_model(model):
    """Create CAM model from the base model"""
    cam_model = Model(inputs=model.input,
                     outputs=(model.layers[-3].output, model.layers[-1].output))
    gap_weights = model.layers[-1].get_weights()[0]
    return cam_model, gap_weights

def preprocess_image(img):
    """Preprocess an already loaded image"""
    if img is None:
        return None, None
        
    # Convert to RGB if needed
    if len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.shape[2] == 3 and img.dtype == np.uint8:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    original_img = img.copy()
    
    # Resize and normalize
    img_resized = cv2.resize(img, (IM_SIZE, IM_SIZE))
    img_normalized = img_resized / 255.0
    
    return np.expand_dims(img_normalized, axis=0), original_img

def create_heatmap(cam_output, img, alpha=0.5):
    """Create a properly scaled heatmap overlay"""
    # Normalize CAM output to 0-1
    cam_output = np.maximum(cam_output, 0)
    cam_output = cam_output / np.max(cam_output) if np.max(cam_output) > 0 else cam_output
    
    # Resize CAM to match image dimensions
    heatmap = cv2.resize(cam_output, (img.shape[1], img.shape[0]))
    
    # Apply colormap - use cv2's colormap for consistency
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend original image with heatmap
    superimposed = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    
    return superimposed, heatmap

def show_cam(original_img, features, predictions, gap_weights, class_idx=1, image_name=None):
    """Generate and display an enhanced, beautiful Class Activation Map visualization"""
    # Set the style for a modern, clean look
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Set custom font properties for a more polished look
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
    # Get the last convolutional layer features
    features_for_img = features[0]
    
    # Get the class weights for the specified class
    class_activation_weights = gap_weights[:, class_idx]
    
    # Create the CAM by doing a weighted sum of activation maps
    cam_output = np.dot(features_for_img, class_activation_weights)
    
    # Create heatmap and overlay
    superimposed_img, heatmap = create_heatmap(cam_output, original_img, alpha=0.6)
    
    # Create custom color map with improved aesthetics
    fire_colors = plt.cm.get_cmap('inferno')
    fire_colormap = LinearSegmentedColormap.from_list('inferno', fire_colors(np.linspace(0, 1, 256)))
    
    # Create a figure with custom layout
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor='#f8f8f8')
    
    # Use GridSpec for more control over layout
    gs = gridspec.GridSpec(2, 6, height_ratios=[4, 1])
    
    # Add a subtle background gradient to the figure
    fig.patch.set_alpha(0.8)
    
    # Original image with enhanced border
    ax1 = plt.subplot(gs[0, 0:2])
    ax1.imshow(original_img)
    ax1.set_title('Original Image', fontweight='bold', pad=15)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_color('#333333')
        spine.set_linewidth(2)
    
    # Heatmap with custom colormap
    ax2 = plt.subplot(gs[0, 2:4])
    heatmap_plot = ax2.imshow(heatmap, cmap=fire_colormap)
    ax2.set_title('Activation Heatmap', fontweight='bold', pad=15)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_color('#333333')
        spine.set_linewidth(2)
    
    # Overlay visualization
    ax3 = plt.subplot(gs[0, 4:6])
    ax3.imshow(superimposed_img)
    ax3.set_title('Overlay Visualization', fontweight='bold', pad=15)
    ax3.set_xticks([])
    ax3.set_yticks([])
    for spine in ax3.spines.values():
        spine.set_visible(True)
        spine.set_color('#333333')
        spine.set_linewidth(2)
    
    # Add colorbar with better styling
    cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=fire_colormap), cax=cbar_ax)
    cbar.set_label('Activation Intensity', fontweight='bold', labelpad=15)
    cbar.outline.set_linewidth(1)
    cbar.outline.set_edgecolor('#333333')
    
    # Prediction visualization with beautiful gauge charts
    # Create a horizontal bar chart for the predictions
    ax4 = plt.subplot(gs[1, 1:5])
    
    # Bar chart for prediction probabilities
    classes = CLASS_NAMES
    probabilities = predictions[0]
    
    # Define colors based on confidence (red for wildfire, blue for no wildfire)
    colors = ['#3498db', '#e74c3c']  # Blue for nowildfire, Red for wildfire
    
    # Create horizontal bars with rounded corners
    bars = ax4.barh(classes, probabilities, color=colors, height=0.5, alpha=0.8)
    
    # Add percentage labels inside the bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_x_pos = width / 2 if width > 0.25 else width + 0.03
        label_alignment = 'center' if width > 0.25 else 'left'
        color = 'white' if width > 0.25 else 'black'
        ax4.text(label_x_pos, bar.get_y() + bar.get_height()/2, 
                 f'{probabilities[i]*100:.1f}%', 
                 va='center', ha=label_alignment, color=color,
                 fontweight='bold', fontsize=12)
    
    # Add a vertical line at 0.5 for reference
    ax4.axvline(x=0.5, color='#7f8c8d', linestyle='--', alpha=0.5)
    
    # Add title to the prediction chart
    ax4.set_title('Classification Probabilities', fontweight='bold', pad=15)
    
    # Set x-axis limits and labels
    ax4.set_xlim(0, 1.0)
    ax4.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax4.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    
    # Adjust spines for cleaner look
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['left'].set_linewidth(1.5)
    ax4.spines['bottom'].set_linewidth(1.5)
    
    # Add a decision indicator
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    decision_text = f"Prediction: {predicted_class.upper()} ({confidence:.1f}% confidence)"
    
    # Add a banner at the top with the decision
    banner_color = '#e74c3c' if predicted_class == 'wildfire' else '#3498db'
    banner_width = 0.90  # Full width banner since we don't have real tag
    banner_ax = fig.add_axes([0.05, 0.92, banner_width, 0.06])
    banner_ax.set_facecolor(banner_color)
    banner_ax.text(0.5, 0.5, decision_text, 
                  ha='center', va='center', color='white', 
                  fontsize=16, fontweight='bold')
    banner_ax.set_xticks([])
    banner_ax.set_yticks([])
    
    # Add image name as footer
    footer_ax = fig.add_axes([0.05, 0.01, 0.9, 0.03])
    footer_ax.axis('off')
    if image_name:
        footer_ax.text(0.5, 0.5, f"Image: {image_name}", 
                      ha='center', va='center', color='#555555', fontsize=10)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 0.9, 0.9])
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    # Convert buffer to PIL Image
    from PIL import Image
    img = Image.open(buf)
    
    return img

def predict_and_visualize(image, model_path):
    """Main function to predict and visualize wildfire detection results"""
    # Check if model path exists and load model
    if not os.path.isfile(model_path):
        from PIL import Image
        # Create a black placeholder image with error message
        placeholder = Image.new('RGB', (800, 600), color=(50, 50, 50))
        return placeholder, "Error: Model file not found at specified path"
    
    try:
        # Load model
        model = load_model_and_weights(model_path)
        if model is None:
            from PIL import Image
            placeholder = Image.new('RGB', (800, 600), color=(50, 50, 50))
            return placeholder, "Error: Failed to load model"
            
        cam_model, gap_weights = prepare_cam_model(model)
        
        # Process image
        image_tensor, original_img = preprocess_image(image)
        if image_tensor is None:
            from PIL import Image
            placeholder = Image.new('RGB', (800, 600), color=(50, 50, 50))
            return placeholder, "Error: Failed to process image"
        
        # Get predictions and features
        features, predictions = cam_model.predict(image_tensor)
        
        # Generate result text
        result_text = f"Prediction Results:\n"
        result_text += f"No Wildfire: {predictions[0][0]*100:.2f}%\n"
        result_text += f"Wildfire: {predictions[0][1]*100:.2f}%\n\n"
        
        # Decision with confidence
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0]) * 100
        result_text += f"Decision: {predicted_class.upper()} detected with {confidence:.1f}% confidence"
        
        # Get image name if possible
        image_name = "Uploaded Image"
        
        # Generate visualization
        viz_img = show_cam(original_img, features, predictions, gap_weights, 
                         class_idx=(1 if predicted_class == 'wildfire' else 0), 
                         image_name=image_name)
        
        # Return the visualization and results text
        return viz_img, result_text
        
    except Exception as e:
        import traceback
        from PIL import Image, ImageDraw, ImageFont
        
        # Create an error image
        error_img = Image.new('RGB', (800, 600), color=(50, 50, 50))
        draw = ImageDraw.Draw(error_img)
        
        # Add error text
        error_msg = f"Error during prediction: {str(e)}"
        draw.text((20, 20), "Error occurred", fill=(255, 50, 50))
        draw.text((20, 60), error_msg, fill=(255, 255, 255))
        
        # Full traceback for debugging
        full_error = f"Error during prediction: {str(e)}\n{traceback.format_exc()}"
        
        return error_img, full_error

def create_gradio_interface():
    """Create and launch the Gradio interface"""
    # Define the model path input
    model_path_input = gr.Textbox(
        label="Model Path", 
        placeholder="Enter the full path to your saved model (.h5 file)",
        value="../saved_models/custom_best_model.h5"
    )
    
    # Define the interface
    iface = gr.Interface(
        fn=predict_and_visualize,
        inputs=[
            gr.Image(type="numpy", label="Upload Image"), 
            model_path_input
        ],
        outputs=[
            gr.Image(type="pil", label="Visualization"),
            gr.Textbox(label="Prediction Results", lines=5)
        ],
        title="Wildfire Detection with XAI Visualization",
        description="""Upload an image to detect wildfires. This tool uses a deep learning model to classify images as containing wildfires or not,
        and provides Class Activation Map (CAM) visualizations to show which regions of the image contributed most to the decision.""",
        examples=[],
        cache_examples=False,
        theme="default"
    )
    
    return iface

# Create and launch the interface
if __name__ == "__main__":
    iface = create_gradio_interface()
    iface.launch(share=False, server_name="0.0.0.0")  # Setting server_name for Kubeflow compatibility