import os
import shutil
import json
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.models import load_model # type: ignore

def prepare_feedback_data():
    """Prepare feedback data for retraining"""
    print("Preparing feedback data...")
    
    # Load existing model config
    with open('titano-N1_config.json', 'r') as f:
        config = json.load(f)
    
    # Create temporary dataset structure
    temp_dir = 'dataset_temp'
    train_dir = os.path.join(temp_dir, 'train')
    val_dir = os.path.join(temp_dir, 'val')
    
    # Copy existing dataset
    shutil.copytree('dataset/train', train_dir, dirs_exist_ok=True)
    shutil.copytree('dataset/val', val_dir, dirs_exist_ok=True)
    
    # Add feedback data to training set
    feedback_dir = 'feedback'
    if os.path.exists(feedback_dir):
        for class_name in os.listdir(feedback_dir):
            if os.path.isdir(os.path.join(feedback_dir, class_name)):
                # Copy feedback images to training directory
                src_dir = os.path.join(feedback_dir, class_name)
                dst_dir = os.path.join(train_dir, class_name)
                os.makedirs(dst_dir, exist_ok=True)
                
                for img in os.listdir(src_dir):
                    if img.endswith(('.jpg', '.jpeg', '.png')):
                        shutil.copy2(
                            os.path.join(src_dir, img),
                            os.path.join(dst_dir, f"feedback_{img}")
                        )
    
    return temp_dir

def prepare_single_image_data(img_path, correct_class):
    """Prepare a single image for incremental learning"""
    print("\nPreparing incremental learning data...")
    
    temp_dir = 'temp_train'
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create class directory
    class_dir = os.path.join(temp_dir, correct_class)
    os.makedirs(class_dir, exist_ok=True)
    
    # Copy image to temp training directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_img_path = os.path.join(class_dir, f'feedback_{timestamp}.jpg')
    shutil.copy2(img_path, new_img_path)
    
    return temp_dir

def quick_retrain(img_path, correct_class, epochs=5):
    """Quickly retrain model on a single image"""
    print("\nStarting quick retraining...")
    
    # Backup current model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = 'model_backups'
    os.makedirs(backup_dir, exist_ok=True)
    model_backup = os.path.join(backup_dir, f'titano-N1_{timestamp}.h5')
    shutil.copy2('titano-N1.h5', model_backup)
    
    try:
        # Load current model
        model = load_model('titano-N1.h5')
        
        # Prepare data generator
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Create temporary training data
        temp_dir = prepare_single_image_data(img_path, correct_class)
        
        # Load class names
        with open('titano-N1_class_names.json', 'r') as f:
            class_names = json.load(f)
        
        # Get model input shape
        with open('titano-N1_config.json', 'r') as f:
            config = json.load(f)
            input_shape = tuple(config.get('input_shape', [224, 224]))
        
        # Set up generator
        train_generator = datagen.flow_from_directory(
            temp_dir,
            target_size=input_shape,
            batch_size=1,
            class_mode='categorical',
            classes=class_names,
            shuffle=True
        )
        
        # Fine-tune with very low learning rate
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Perform quick retraining
        print(f"\nFine-tuning model on new image for {epochs} epochs...")
        model.fit(
            train_generator,
            epochs=epochs,
            steps_per_epoch=10,  # Multiple augmented versions of the same image
            verbose=1
        )
        
        # Save improved model
        model.save('titano-N1.h5')
        print("\nModel updated successfully!")
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print(f"Error during quick retraining: {e}")
        # Restore backup
        shutil.copy2(model_backup, 'titano-N1.h5')
        print("Restored previous model due to error")
        return False
    
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def retrain():
    """Retrain the model with feedback data"""
    # Backup existing model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = 'model_backups'
    os.makedirs(backup_dir, exist_ok=True)
    
    model_backup = os.path.join(backup_dir, f'titano-N1_{timestamp}.h5')
    config_backup = os.path.join(backup_dir, f'titano-N1_config_{timestamp}.json')
    
    shutil.copy2('titano-N1.h5', model_backup)
    shutil.copy2('titano-N1_config.json', config_backup)
    
    # Prepare data for retraining
    temp_dataset = prepare_feedback_data()
    
    try:
        # Import train script and run training
        import train
        print("\nStarting retraining with feedback data...")
        train.train_model(temp_dataset, fine_tune=True)
        
        # Clean up
        shutil.rmtree(temp_dataset)
        print(f"\nModel retrained successfully!")
        print(f"Previous model backed up to: {model_backup}")
        
    except Exception as e:
        print(f"Error during retraining: {e}")
        # Restore backup if training failed
        shutil.copy2(model_backup, 'titano-N1.h5')
        shutil.copy2(config_backup, 'titano-N1_config.json')
        print("Restored previous model due to training error")
    
    finally:
        # Clean up temporary dataset
        if os.path.exists(temp_dataset):
            shutil.rmtree(temp_dataset)

if __name__ == "__main__":
    retrain()
