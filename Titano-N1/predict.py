import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings

# Disable GPU if there are CUDA errors
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

def get_model_input_shape(model):
    """Get input shape from model using multiple methods"""
    try:
        # Get shape from first layer directly
        first_layer = model.layers[0]
        if hasattr(first_layer, 'input_shape'):
            shape = first_layer.input_shape[1:3]
            print(f"Detected input shape from first layer: {shape}")
            return shape
    except:
        pass
    
    try:
        # Try getting shape from model input
        shape = model.input_shape[1:3]
        print(f"Detected input shape from model input: {shape}")
        return shape
    except:
        pass
    
    # If both methods fail, try to load from a saved config file
    try:
        if os.path.exists('titano-N1_config.json'):
            with open('titano-N1_config.json', 'r') as f:
                config = json.load(f)
                shape = tuple(config.get('input_shape', (150, 150)))
                print(f"Loaded input shape from config: {shape}")
                return shape
    except:
        pass
    
    # Default to light model size if everything fails
    print("Warning: Could not detect model input shape, using light model size (150, 150)")
    return (150, 150)

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

model_path = 'titano-N1.h5'
if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' not found. Please train the model first.")
    exit()

try:
    print("Loading model...")
    model = load_model(model_path, compile=False)  # Load without compilation
    input_shape = get_model_input_shape(model)
    
    # Save the detected shape for future use
    try:
        with open('titano-N1_config.json', 'w') as f:
            json.dump({'input_shape': input_shape}, f)
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
        img = image.load_img(img_path, target_size=input_shape)
        x = image.img_to_array(img)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)

        print("Making prediction...")
        preds = model.predict(x, verbose=0)
        predicted_class = class_names[np.argmax(preds)]
        confidence = np.max(preds) * 100
        
        print("\nClass probabilities:")
        for class_name, prob in zip(class_names, preds[0]):
            print(f"{class_name}: {prob*100:.2f}%")
            
        return predicted_class, confidence

    if __name__ == "__main__":
        test_image = input("\nEnter the path to your image: ").strip()
        if not os.path.exists(test_image):
            print(f"Error: Image '{test_image}' not found. Please provide a valid image path.")
        else:
            predicted_class, confidence = predict_image(test_image)
            print(f"\nFinal Prediction:")
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2f}%")

except Exception as e:
    print(f"Error during prediction: {e}")
    import traceback
    traceback.print_exc()