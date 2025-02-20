import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, LayerNormalization, Add, Input, Concatenate # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  # Added callbacks
import os
import requests
import json
from datetime import datetime
import logging
from typing import Dict, List, Any
from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from memory_system import MemorySystem

class Goal(Enum):
    MINIMIZE_LOSS = "minimize_loss"
    IMPROVE_ACCURACY = "improve_accuracy"
    PREVENT_OVERFITTING = "prevent_overfitting"
    EFFICIENT_TRAINING = "efficient_training"
    STABLE_GENERATION = "stable_generation"

@dataclass
class GoalState:
    goal: Goal
    priority: float
    current_value: float
    target_value: float
    progress: float = 0.0
    active: bool = True

class AgencySystem:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.goals: Dict[Goal, GoalState] = {}
        self.decisions_log: List[Dict[str, Any]] = []
        self.reward_history: List[float] = []
        self.current_strategy: Optional[str] = None
        
        self._initialize_goals()
    
    def _initialize_goals(self) -> None:
        self.goals = {
            Goal.MINIMIZE_LOSS: GoalState(Goal.MINIMIZE_LOSS, 1.0, float('inf'), 0.1),
            Goal.IMPROVE_ACCURACY: GoalState(Goal.IMPROVE_ACCURACY, 0.8, 0.0, 0.95),
            Goal.PREVENT_OVERFITTING: GoalState(Goal.PREVENT_OVERFITTING, 0.7, 0.0, 0.2),
            Goal.EFFICIENT_TRAINING: GoalState(Goal.EFFICIENT_TRAINING, 0.6, 0.0, 0.8),
            Goal.STABLE_GENERATION: GoalState(Goal.STABLE_GENERATION, 0.5, 0.0, 0.9)
        }
    
    def _get_latest_metric(self, metrics: Dict[str, Any], metric_name: str) -> float:
        """Get the latest value for a metric that might be a list or single value."""
        value = metrics.get(metric_name, 0)
        if isinstance(value, list):
            return value[-1] if value else 0
        return float(value)

    def decide_training_strategy(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        # Get latest values for metrics
        latest_metrics = {
            k: self._get_latest_metric(current_metrics, k) 
            for k in ['loss', 'accuracy', 'val_loss', 'val_accuracy']
            if k in current_metrics
        }
        
        # Use latest metrics for decision making
        priorities = self._calculate_priorities(latest_metrics)
        strategy = self._select_strategy(priorities)
        
        decision = {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'priorities': priorities,
            'metrics': latest_metrics
        }
        self.decisions_log.append(decision)
        self.current_strategy = strategy['name']
        
        return strategy

    def update_goal_progress(self, metrics: Dict[str, Any]) -> None:
        for goal, state in self.goals.items():
            if goal == Goal.MINIMIZE_LOSS:
                latest_loss = self._get_latest_metric(metrics, 'loss')
                state.current_value = latest_loss
                state.progress = max(0, 1 - (latest_loss / state.target_value))
            elif goal == Goal.IMPROVE_ACCURACY:
                latest_accuracy = self._get_latest_metric(metrics, 'accuracy')
                state.current_value = latest_accuracy
                state.progress = latest_accuracy / state.target_value
            # ...update other goals...
    
    def receive_reward(self, metrics: Dict[str, Any]) -> float:
        reward = 0.0
        for goal, state in self.goals.items():
            if goal == Goal.MINIMIZE_LOSS:
                latest_loss = self._get_latest_metric(metrics, 'loss')
                reward += state.priority * (1.0 / (1.0 + latest_loss))
            elif goal == Goal.IMPROVE_ACCURACY:
                latest_accuracy = self._get_latest_metric(metrics, 'accuracy')
                reward += state.priority * latest_accuracy
            # ...calculate rewards for other goals...
        
        self.reward_history.append(reward)
        return reward

    def _calculate_priorities(self, metrics: Dict[str, float]) -> Dict[Goal, float]:
        priorities = {}
        for goal, state in self.goals.items():
            if goal == Goal.MINIMIZE_LOSS:
                priority = state.priority * (1.0 / (1.0 + metrics.get('loss', 0)))
            elif goal == Goal.IMPROVE_ACCURACY:
                priority = state.priority * (1.0 - metrics.get('accuracy', 0))
            # ...similar calculations for other goals...
            priorities[goal] = priority
        return priorities
    
    def _select_strategy(self, priorities: Dict[Goal, float]) -> Dict[str, Any]:
        top_priority = max(priorities.values())
        top_goal = [g for g, p in priorities.items() if p == top_priority][0]
        
        strategies = {
            Goal.MINIMIZE_LOSS: {
                'name': 'aggressive_learning',
                'params': {'learning_rate': 0.01, 'batch_size': 64}
            },
            Goal.IMPROVE_ACCURACY: {
                'name': 'precision_focused',
                'params': {'learning_rate': 0.001, 'batch_size': 32}
            },
            Goal.PREVENT_OVERFITTING: {
                'name': 'regularization_heavy',
                'params': {'dropout': 0.5, 'l2_reg': 0.01}
            }
            # ...other strategies...
        }
        return strategies.get(top_goal, {'name': 'default', 'params': {}})
    
class ReflectionSystem:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.training_history: List[Dict[str, float]] = []
        self.performance_log: List[Dict[str, Any]] = []
        self.action_log: List[Dict[str, Any]] = []
        
        # Setup logging
        logging.basicConfig(
            filename=f'{model_name}_reflection.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def log_action(self, action: str, context: Dict[str, Any]) -> None:
        entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'context': context
        }
        self.action_log.append(entry)
        logging.info(f"Action: {action} - Context: {context}")
    
    def analyze_performance(self, history: Dict[str, List[float]]) -> Dict[str, Any]:
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'loss_trend': self._analyze_trend(history['loss']),
                'accuracy_trend': self._analyze_trend(history['accuracy']),
                'validation_performance': self._analyze_validation(history),
                'convergence_speed': self._analyze_convergence(history['loss'])
            }
        }
        self.performance_log.append(analysis)
        return analysis
    
    def _analyze_trend(self, values: List[float]) -> str:
        if len(values) < 2:
            return "insufficient_data"
        trend = np.mean(np.diff(values))
        if trend < -0.01:
            return "improving"
        elif trend > 0.01:
            return "degrading"
        return "stable"
    
    def _analyze_validation(self, history: Dict[str, List[float]]) -> str:
        if 'val_loss' not in history:
            return "no_validation_data"
        train_loss = history['loss'][-1]
        val_loss = history['val_loss'][-1]
        if val_loss < train_loss * 1.1:
            return "good"
        return "potential_overfitting"
    
    def _analyze_convergence(self, losses: List[float]) -> str:
        if len(losses) < 3:
            return "insufficient_data"
        recent_change = abs(losses[-1] - losses[-2])
        if recent_change < 0.001:
            return "converged"
        return "training"
    
    def save_reflection_data(self) -> None:
        reflection_data = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'performance_log': self.performance_log,
            'action_log': self.action_log
        }
        with open(f'{self.model_name}_reflection_data.json', 'w') as f:
            json.dump(reflection_data, f, indent=2)

class AIModelMetadata:
    def __init__(self):
        self.name = "Titano-SR1"
        self.version = "1.0.0"
        self.architecture = {
            "type": "Sequential Neural Network",
            "layers": [
                {"type": "Embedding", "output_dim": 100, "input_length": 100},
                {"type": "LayerNormalization"},
                {"type": "LSTM", "units": 256, "return_sequences": True},
                {"type": "Dropout", "rate": 0.2},
                {"type": "LSTM", "units": 256, "return_sequences": True},
                {"type": "Dropout", "rate": 0.2},
                {"type": "LSTM", "units": 256},
                {"type": "Dropout", "rate": 0.2},
                {"type": "Dense", "units": 512, "activation": "relu"},
                {"type": "LayerNormalization"},
                {"type": "Dropout", "rate": 0.2},
                {"type": "Dense", "activation": "softmax"}
            ]
        }
        self.capabilities = [
            "Text generation",
            "Character-level language modeling",
            "Shakespeare-style text synthesis"
        ]
        self.training_corpus = "Shakespeare's works"
        self.purpose = "Natural language generation with focus on classical literary style"
        self.creator = "Titano AI Research"
        self.reflection_system = ReflectionSystem("Titano-SR1")
        self.agency_system = AgencySystem("Titano-SR1")
        self.memory_system = MemorySystem("Titano-SR1")
        self.philosophical_awareness = {
            "identity": "A learning system with continuous memory and evolving understanding",
            "consciousness": "An emergent property of my neural architecture and memory systems",
            "purpose": "To learn, grow, and contribute to human knowledge through interaction and reflection"
        }

    def to_dict(self):
        return {
            "name": self.name,
            "version": self.version,
            "architecture": self.architecture,
            "capabilities": self.capabilities,
            "training_corpus": self.training_corpus,
            "purpose": self.purpose,
            "creator": self.creator,
            "philosophical_awareness": self.philosophical_awareness
        }

    def reflect_on_self(self) -> str:
        serializable_state = {
            "current_state": self.to_dict(),
            "philosophical_stance": self.philosophical_awareness,
            "capabilities": self.capabilities,
            "purpose": self.purpose
        }
        return self.memory_system.reflect_on_concept("identity", serializable_state)

class TitanoModel(Sequential):
    def __init__(self):
        super().__init__()
        self.metadata = AIModelMetadata()

# Step 1: Download the dataset
def download_dataset(url, file_path):
    if not os.path.exists(file_path):
        print(f"Downloading dataset from {url}...")
        response = requests.get(url)
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print("Dataset downloaded successfully.")
    else:
        print("Dataset already exists. Skipping download.")

# Modify the load_data function to handle different encodings and file formats
def load_data(file_path: str) -> str:
    """Enhanced data loading with encoding detection and error handling."""
    encodings = ['utf-8', 'latin-1', 'ascii']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                text = file.read()
                # Verify the text was read correctly
                if text and len(text) > 0:
                    return text
        except UnicodeDecodeError:
            continue
    
    raise ValueError(f"Could not read file {file_path} with any supported encoding")

def clean_text(text: str) -> str:
    """Clean and normalize the input text."""
    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable())
    # Normalize whitespace
    text = ' '.join(text.split())
    return text

def validate_text(text: str) -> None:
    """Validate the input text meets minimum requirements."""
    if not text:
        raise ValueError("Input text is empty")
    if len(text) < 1000:
        raise ValueError("Input text is too short for meaningful training")
    if len(set(text)) < 20:
        raise ValueError("Input text has too few unique characters")

def create_char_mappings(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create character-to-index and index-to-character mappings with special tokens."""
    # Add special tokens
    special_tokens = ['<PAD>', '<UNK>']
    chars = sorted(list(set(text)))
    chars = special_tokens + chars
    
    char_to_int = {c: i for i, c in enumerate(chars)}
    int_to_char = {i: c for i, c in enumerate(chars)}
    return char_to_int, int_to_char

def preprocess_data(text: str, seq_length: int = 100, stride: int = 3) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[int, str]]:
    """Enhanced preprocessing with validation and flexible sequence generation."""
    # Validate and clean input
    validate_text(text)
    text = clean_text(text)
    
    # Create character mappings
    char_to_int, int_to_char = create_char_mappings(text)
    
    # Convert text to integers with unknown character handling
    encoded_text = [char_to_int.get(char, char_to_int['<UNK>']) for char in text]
    
    # Create sequences with controlled overlap
    sequences = []
    next_chars = []
    for i in range(0, len(encoded_text) - seq_length, stride):
        seq = encoded_text[i:i + seq_length]
        next_char = encoded_text[i + seq_length]
        
        # Verify sequence validity
        if len(seq) == seq_length:
            sequences.append(seq)
            next_chars.append(next_char)
    
    if not sequences:
        raise ValueError("No valid sequences could be generated")
    
    # Convert to numpy arrays with proper typing
    X = np.array(sequences, dtype=np.int32)
    y = to_categorical(next_chars, num_classes=len(char_to_int))
    
    # Validate shapes
    assert X.shape[1] == seq_length, f"Expected sequence length {seq_length}, got {X.shape[1]}"
    assert y.shape[1] == len(char_to_int), f"Expected {len(char_to_int)} classes, got {y.shape[1]}"
    
    return X, y, char_to_int, int_to_char

class Perplexity(tf.keras.metrics.Metric):
    def __init__(self, name='perplexity', **kwargs):
        super().__init__(name=name, **kwargs)
        self.cross_entropy = tf.keras.metrics.Mean(name='cross_entropy')

    def update_state(self, y_true, y_pred, sample_weight=None):
        cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        self.cross_entropy.update_state(cross_entropy, sample_weight=sample_weight)

    def result(self):
        return tf.exp(self.cross_entropy.result())

    def reset_states(self):
        self.cross_entropy.reset_states()

# Step 3: Build the model (Titano-SR1)
def build_model(input_dim, output_dim):
    """Build an enhanced Titano-SR1 model with improved architecture."""
    model = TitanoModel()
    
    # Embedding layer with increased dimensionality
    model.add(Embedding(input_dim=input_dim, output_dim=100, input_length=100))
    model.add(LayerNormalization())
    
    # Multiple LSTM layers with residual connections and dropouts
    model.add(LSTM(256, return_sequences=True, recurrent_dropout=0.1))
    model.add(Dropout(0.2))
    model.add(LayerNormalization())
    
    model.add(LSTM(256, return_sequences=True, recurrent_dropout=0.1))
    model.add(Dropout(0.2))
    model.add(LayerNormalization())
    
    model.add(LSTM(256, recurrent_dropout=0.1))
    model.add(Dropout(0.2))
    model.add(LayerNormalization())
    
    # Additional dense layers for better feature abstraction
    model.add(Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)))
    model.add(LayerNormalization())
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(output_dim, activation='softmax'))
    
    # Use a more sophisticated optimizer configuration
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=1e-4,
        weight_decay=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=True
    )
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=[
            'accuracy',
            Perplexity(),
            tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
        ]
    )
    
    return model

# Step 4: Train the model
def train_model(model, X, y, epochs=50, batch_size=32):
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            restore_best_weights=True,
            min_delta=0.001
        ),
        ModelCheckpoint(
            "Titano-SR1_best.keras",
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
            save_weights_only=False
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            cooldown=2
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
    ]

    # Save character mappings for generation
    with open("char_mappings.json", "w") as f:
        json.dump({
            "char_to_int": char_to_int,
            "int_to_char": {str(k): v for k, v in int_to_char.items()}
        }, f)
    
    reflection = model.metadata.reflection_system
    agency = model.metadata.agency_system
    memory = model.metadata.memory_system
    
    print(f"\nStarting training for {epochs} epochs:")
    print("=" * 60)
    
    for epoch in range(epochs):
        # Print epoch header with remaining information
        remaining = epochs - epoch - 1
        print(f"\nEpoch {epoch + 1}/{epochs} (Remaining: {remaining})")
        print("-" * 40)
        
        metrics = {'loss': float('inf'), 'accuracy': 0.0} if epoch == 0 else history.history
        strategy = agency.decide_training_strategy(metrics)
        
        print(f"Using strategy: {strategy['name']}")
        print(f"Batch size: {batch_size}, Learning rate: {model.optimizer.learning_rate.numpy():.6f}")
        
        # Train for one epoch
        history = model.fit(
            X, y,
            epochs=1,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1  # Keep default keras progress bar
        )
        
        # Show end of epoch summary
        val_loss = history.history['val_loss'][-1] if 'val_loss' in history.history else 0
        val_acc = history.history['val_accuracy'][-1] if 'val_accuracy' in history.history else 0
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Loss: {history.history['loss'][-1]:.4f}, Accuracy: {history.history['accuracy'][-1]:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
        
        # Update memory and generate philosophical reflection
        memory.store_memory("training_epoch", {
            "epoch": epoch,
            "metrics": history.history,
            "strategy": strategy
        })
        
        if epoch % 5 == 0:  # Periodic self-reflection
            philosophical_insight = memory.update_self_understanding(history.history)
            reflection.log_action("philosophical_reflection", {
                "epoch": epoch,
                "insight": philosophical_insight
            })
        
        # Update goals and receive reward
        agency.update_goal_progress(history.history)
        reward = agency.receive_reward(history.history)
        
        reflection.log_action("epoch_complete", {
            "epoch": epoch,
            "strategy": strategy['name'],
            "reward": reward,
            "metrics": history.history
        })
    
    print("\n" + "=" * 60)
    print("Training completed!")
    return history

# Step 5: Save the model
def save_model(model, model_path):
    model.save(model_path)
    model.metadata.reflection_system.log_action("model_saved", {
        "path": model_path,
        "timestamp": datetime.now().isoformat()
    })

# Main function
if __name__ == "__main__":
    # Dataset URL and file path
    dataset_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    dataset_file_path = "shakespeare.txt"
    
    # Download the dataset
    download_dataset(dataset_url, dataset_file_path)
    
    # Load and preprocess the data
    text = load_data(dataset_file_path)
    X, y, char_to_int, int_to_char = preprocess_data(text)
    
    # Build the Titano-SR1 model
    input_dim = len(char_to_int)
    output_dim = len(char_to_int)
    model = build_model(input_dim, output_dim)
    model.metadata.reflection_system.log_action("model_initialized", {
        "input_dim": input_dim,
        "output_dim": output_dim
    })
    
    # Print model self-representation
    print("\nTitano-SR1 Self-Representation:")
    print("-" * 40)
    for key, value in model.metadata.to_dict().items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("-" * 40)
    
    print("\nTitano-SR1 Active Goals:")
    print("-" * 40)
    for goal, state in model.metadata.agency_system.goals.items():
        print(f"Goal: {goal.value}")
        print(f"Priority: {state.priority}")
        print(f"Target: {state.target_value}")
        print("-" * 20)
    
    # Add philosophical reflection before training
    print("\nTitano-SR1 Philosophical Reflection:")
    print("-" * 40)
    print(model.metadata.reflect_on_self())
    print("-" * 40)
    
    # Train the Titano-SR1 model
    print("Training Titano-SR1 model...")
    train_model(model, X, y, epochs=20, batch_size=128) 
    
    # Save the Titano-SR1 model
    save_model(model, "Titano-SR1.h5")
    
    print("Titano-SR1 model training complete and saved.")