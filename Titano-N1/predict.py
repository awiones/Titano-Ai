import os
import json
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import tensorflow as tf
import warnings
import shutil
from datetime import datetime
warnings.filterwarnings('ignore')  # Suppress warnings

# Disable GPU if there are CUDA errors
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

def get_model_input_shape(model):
    """Get input shape from model using multiple methods"""
    try:
        # Get shape from first layer
        first_layer = model.layers[0]
        if hasattr(first_layer, 'input_shape'):
            shape = first_layer.input_shape[1:3]
            channels = first_layer.input_shape[3]  # Get expected channels
            print(f"Detected input shape from first layer: {shape}, channels: {channels}")
            return shape, channels
    except:
        pass
    
    try:
        # Try getting shape from model input
        shape = model.input_shape[1:3]
        channels = model.input_shape[3]  # Get expected channels
        print(f"Detected input shape from model input: {shape}, channels: {channels}")
        return shape, channels
    except:
        pass
    
    # Default to light model size if everything fails
    print("Warning: Could not detect model input shape, using default (150, 150, 1)")
    return (150, 150), 1

def load_class_names():
    if os.path.exists('titano-N1_class_names.json'):
        with open('titano-N1_class_names.json', 'r') as f:
            return json.load(f)
    elif os.path.exists('dataset/train'):
        classes = [d for d in os.listdir('dataset/train') 
                  if not d.startswith('.') and os.path.isdir(os.path.join('dataset/train', d))]
        return sorted(classes)
    else:
        raise ValueError("No class names found. Please train the model first.")

def log_incorrect_prediction(img_path, predicted_class, actual_class):
    """Log incorrect predictions for future retraining"""
    feedback_dir = 'feedback'
    os.makedirs(feedback_dir, exist_ok=True)
    
    # Create directories for the actual class
    actual_dir = os.path.join(feedback_dir, actual_class)
    os.makedirs(actual_dir, exist_ok=True)
    
    # Copy the image to feedback directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    img_name = f"{predicted_class}_to_{actual_class}_{timestamp}.jpg"
    dst_path = os.path.join(actual_dir, img_name)
    
    shutil.copy2(img_path, dst_path)
    
    # Log the prediction details
    with open(os.path.join(feedback_dir, 'feedback_log.txt'), 'a') as f:
        f.write(f"{timestamp},{img_path},{predicted_class},{actual_class}\n")

model_path = 'titano-N1.h5'
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found. Please train the model first.")
    exit()

try:
    print("Loading model...")
    model = load_model(model_path, compile=False)  # Load without compilation
    input_shape, channels = get_model_input_shape(model)
    
    # Load model configuration
    try:
        with open('titano-N1_config.json', 'r') as f:
            config = json.load(f)
            grayscale = config.get('grayscale', channels == 1)
            is_text_model = config.get('type') == 'text'
            print(f"Model type: {'Text Recognition' if is_text_model else 'Image Classification'}")
            print(f"Grayscale mode: {grayscale}")
    except:
        grayscale = channels == 1
        is_text_model = False
    
    # Save the detected configuration
    try:
        with open('titano-N1_config.json', 'w') as f:
            json.dump({
                'input_shape': input_shape,
                'channels': channels,
                'grayscale': grayscale,
                'type': 'text' if is_text_model else 'image'
            }, f)
    except:
        pass
        
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print(f"Model expects input shape: {input_shape}")
    class_names = load_class_names()
    print(f"Loaded {len(class_names)} classes: {class_names}")

    def predict_image(img_path):
        if not os.path.exists(img_path):
            print(f"Error: Image file '{img_path}' not found.")
            return None

        print(f"Processing image to size {input_shape}...")
        
        # For text/captcha images, always use grayscale
        if is_text_model:
            print("Text recognition mode: Converting to grayscale")
            color_mode = 'grayscale'
        else:
            color_mode = 'grayscale' if grayscale else 'rgb'
            
        # Load and preprocess image
        img = image.load_img(img_path, 
                           target_size=input_shape,
                           color_mode=color_mode)
        
        x = image.img_to_array(img)
        
        # Ensure correct number of channels
        if is_text_model or grayscale:
            if len(x.shape) == 3 and x.shape[-1] != 1:
                x = np.mean(x, axis=-1, keepdims=True)
        
        # Additional preprocessing for text images
        if is_text_model:
            # Enhance contrast
            x = ((x - x.min()) / (x.max() - x.min()) * 255).astype(np.uint8)
            # Convert back to float and normalize
            x = x.astype(np.float32) / 255.0
        else:
            x = x / 255.0
        
        # Add batch dimension
        x = np.expand_dims(x, axis=0)
        
        print(f"Final input shape: {x.shape}")
        
        # Make prediction
        preds = model.predict(x, verbose=0)
        predicted_class = class_names[np.argmax(preds)]
        confidence = np.max(preds) * 100
        
        print("\nClass probabilities:")
        for class_name, prob in zip(class_names, preds[0]):
            print(f"{class_name}: {prob*100:.2f}%")
            
        return predicted_class, confidence

    if __name__ == "__main__":
        last_incorrect_image = None
        
        while True:
            test_image = input("\nEnter the path to your image (or 'q' to quit): ").strip()
            if test_image.lower() == 'q':
                break
                
            if not os.path.exists(test_image):
                print(f"Error: Image '{test_image}' not found. Please provide a valid image path.")
                continue
                
            predicted_class, confidence = predict_image(test_image)
            print(f"\nFinal Prediction:")
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2f}%")
            
            # Get feedback
            feedback = input("\nIs this prediction accurate? (Y/n): ").strip().lower()
            if feedback == 'n':
                print("\nAvailable classes:", ', '.join(class_names))
                actual_class = input("What is the correct class? ").strip()
                
                if actual_class in class_names:
                    # Log the incorrect prediction
                    log_incorrect_prediction(test_image, predicted_class, actual_class)
                    print(f"\nFeedback saved.")
                    
                    # Perform immediate retraining
                    print("\nPerforming quick model update...")
                    import improve_model
                    if improve_model.quick_retrain(test_image, actual_class):
                        print("\nModel updated! Let's verify the improvement...")
                        
                        # Reload the model
                        model = load_model('titano-N1.h5', compile=False)
                        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                        
                        # Try the prediction again
                        print("\nRetrying prediction with improved model...")
                        new_predicted_class, new_confidence = predict_image(test_image)
                        print(f"\nNew Prediction:")
                        print(f"Predicted Class: {new_predicted_class}")
                        print(f"Confidence: {new_confidence:.2f}%")
                    
                else:
                    print(f"Invalid class name. Please use one of: {', '.join(class_names)}")
            
            print("\nReady for next prediction...")

except Exception as e:
    print(f"Error during prediction: {e}")
    import traceback
    traceback.print_exc()