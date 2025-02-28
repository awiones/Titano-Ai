import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import re
import json
from collections import Counter
import math
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, RandomSampler
import torch.multiprocessing as mp

# Add Google Colab drive import at the top
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enhanced tokenizer class
class SimpleTokenizer:
    def __init__(self, vocab_size=30000):  # Increased default vocabulary size
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.unk_token = "<UNK>"
        self.pad_token = "<PAD>"
        self.eos_token = "<EOS>"
        self.bos_token = "<BOS>"
        
        # Add special tokens
        self.word_to_idx[self.pad_token] = 0
        self.word_to_idx[self.unk_token] = 1
        self.word_to_idx[self.bos_token] = 2
        self.word_to_idx[self.eos_token] = 3
        
        # Update idx_to_word mapping
        for word, idx in self.word_to_idx.items():
            self.idx_to_word[idx] = word
            
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
    
    def train_from_texts(self, texts):
        """Build vocabulary from list of texts"""
        # Tokenize all texts
        all_words = []
        for text in tqdm(texts, desc="Building vocabulary"):
            words = self._tokenize(text)
            all_words.extend(words)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Select top words for vocabulary
        vocab_size = min(self.vocab_size, len(word_counts) + 4)  # +4 for special tokens
        most_common = word_counts.most_common(vocab_size - 4)  # -4 for special tokens
        
        # Add words to vocabulary
        for word, _ in most_common:
            if word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        
        print(f"Vocabulary size: {len(self.word_to_idx)}")
    
    def _tokenize(self, text):
        """Improved word tokenization"""
        # Keep apostrophes and perform enhanced cleaning
        text = re.sub(r"[^a-zA-Z0-9'\s]", ' ', text.lower())
        # Handle contractions better (e.g., don't, can't)
        text = re.sub(r"(\w)'(\w)", r"\1\2", text)  # Replace word'word with wordword
        # Collapse multiple spaces and trim
        text = re.sub(r'\s+', ' ', text).strip()
        return text.split()
    
    def encode(self, text, add_special_tokens=True, max_length=None, padding=False):
        """Convert text to token IDs"""
        words = self._tokenize(text)
        
        # Add special tokens if requested
        if add_special_tokens:
            words = [self.bos_token] + words + [self.eos_token]
        
        # Convert words to ids
        ids = [self.word_to_idx.get(word, self.unk_token_id) for word in words]
        
        # Truncate if needed
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        
        # Add padding if requested
        if padding and max_length is not None:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        
        return torch.tensor(ids)
    
    def decode(self, ids, skip_special_tokens=True):
        """Convert token IDs back to text"""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
            
        words = []
        for idx in ids:
            word = self.idx_to_word.get(idx, self.unk_token)
            if skip_special_tokens and word in [self.pad_token, self.unk_token, self.bos_token, self.eos_token]:
                continue
            words.append(word)
        
        return " ".join(words)
    
    def get_vocab_size(self):
        return len(self.word_to_idx)
    
    def save(self, path):
        with open(path, 'w') as f:
            json.dump({
                'word_to_idx': self.word_to_idx,
                'idx_to_word': {int(k): v for k, v in self.idx_to_word.items()}  # Convert keys to strings for JSON
            }, f)
    
    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
            self.word_to_idx = data['word_to_idx']
            self.idx_to_word = {int(k): v for k, v in data['idx_to_word'].items()}  # Convert keys back to integers

# Enhanced model architecture with residual connections and layer normalization
class TitanoSR1(nn.Module):
    def __init__(self, vocab_size, embedding_dim=384, hidden_dim=768, num_layers=4, dropout=0.1):
        super(TitanoSR1, self).__init__()
        # Ensure valid dimensions
        assert vocab_size > 0, "Vocabulary size must be positive"
        assert embedding_dim > 0, "Embedding dimension must be positive"
        assert hidden_dim > 0, "Hidden dimension must be positive"
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        
        # LSTM with proper initialization
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Adjusted dimensions for bidirectional LSTM
        lstm_output_dim = hidden_dim * 2
        
        # Layer normalization with correct dimensions
        self.layer_norm = nn.LayerNorm(lstm_output_dim)
        
        # Feed-forward layers with proper dimensions
        self.ff1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # Output projection with correct dimensions
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize weights with proper dimensions
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        # Initialize embedding
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
        
        # Initialize LSTM
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Initialize linear layers
        nn.init.xavier_uniform_(self.ff1.weight)
        nn.init.zeros_(self.ff1.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x):
        # Ensure input has correct shape
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # Get embeddings
        embeds = self.embedding(x)  # Shape: [batch_size, seq_length, embedding_dim]
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(embeds)  # Shape: [batch_size, seq_length, hidden_dim*2]
        
        # Apply layer normalization
        normalized = self.layer_norm(lstm_out)  # Shape maintains
        
        # Feed-forward network
        ff_out = self.ff1(normalized)  # Shape: [batch_size, seq_length, hidden_dim]
        ff_out = self.activation(ff_out)
        ff_out = self.dropout(ff_out)
        
        # Project to vocabulary size
        output = self.fc(ff_out)  # Shape: [batch_size, seq_length, vocab_size]
        
        return output

# Custom dataset with better preprocessing for the natural reasoning data
class NaturalReasoningDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_length=64):  # Increased sequence length
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Filter and clean texts
        self.texts = [self._clean_text(text) for text in texts if self._is_valid_text(text)]
        self.vocab_size = tokenizer.get_vocab_size()
        
        print(f"Dataset created with {len(self.texts)} valid examples")
        
    def _is_valid_text(self, text):
        """Filter low-quality texts"""
        if not text or len(text) < 30:  # Minimum length
            return False
        
        # Check if text has enough words
        words = text.split()
        if len(words) < 10:
            return False
            
        # Check for excessive repetition (potential garbage text)
        word_set = set(words)
        if len(word_set) < len(words) * 0.4:  # At least 40% unique words
            return False
            
        return True
        
    def _clean_text(self, text):
        """Clean text for better quality"""
        # Replace excessive newlines
        text = re.sub(r'\n+', ' ', text)
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Replace multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize the text
        token_ids = self.tokenizer.encode(
            text, 
            max_length=self.seq_length + 1,
            padding=True
        )
        
        # If we have fewer tokens than seq_length + 1, pad
        if len(token_ids) < self.seq_length + 1:
            token_ids = torch.cat([
                token_ids, 
                torch.zeros(self.seq_length + 1 - len(token_ids), dtype=torch.long)
            ])
        
        input_ids = token_ids[:self.seq_length]
        target_ids = token_ids[1:self.seq_length+1]
        
        return input_ids, target_ids

# Enhanced text generation function with better sampling strategies
def generate_text(model, tokenizer, seed_text="The reason is", max_length=100, 
                 temperature=0.8, top_k=50, top_p=0.85, repetition_penalty=1.2):
    # Store the original training state
    was_training = model.training
    
    # Set model to eval mode
    model.eval()
    
    tokens = tokenizer.encode(seed_text)
    tokens = tokens.to(device)
    
    generated = tokens.tolist()
    past_tokens = set(generated)
    
    try:
        with torch.no_grad():
            for _ in range(max_length):
                # Only use the last seq_length tokens if we have too many
                if len(tokens) > 64:
                    input_tokens = tokens[-64:]
                else:
                    input_tokens = tokens
                
                # Forward pass
                input_tokens = input_tokens.unsqueeze(0)
                outputs = model(input_tokens)
                
                # Get the predictions for the next token
                next_token_logits = outputs[0, -1, :].clone()
                
                # Apply repetition penalty (reduce probability of tokens we've seen before)
                for token_id in past_tokens:
                    next_token_logits[token_id] /= repetition_penalty
                    
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                    
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Add the predicted token to the sequence
                tokens = torch.cat([tokens, next_token], dim=0)
                past_tokens.add(next_token.item())
                
                # Stop if we predict the EOS token
                if next_token.item() == tokenizer.eos_token_id:
                    break
    
    finally:
        # Restore the original training state
        if was_training:
            model.train()
        
    # Decode the tokens to text
    output_text = tokenizer.decode(tokens)
    return output_text

def check_model_exists():
    return os.path.exists("titano_sr1_model.pt") and os.path.exists("titano_sr1_tokenizer.json")

def load_existing_model():
    tokenizer = SimpleTokenizer()
    tokenizer.load("titano_sr1_tokenizer.json")
    
    checkpoint = torch.load("titano_sr1_model.pt", map_location=device)
    model = TitanoSR1(
        vocab_size=checkpoint['vocab_size'],
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint.get('dropout', 0.1)  # Default to 0.1 if not in checkpoint
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, tokenizer

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*50)
    print(f"{text:^50}")
    print("="*50 + "\n")

def print_menu(options):
    """Print a numbered menu with options"""
    for idx, option in enumerate(options, 1):
        print(f"{idx}. {option}")

def get_choice(prompt, valid_range):
    """Get a valid choice from user"""
    while True:
        try:
            choice = int(input(f"\n{prompt}: ").strip())
            if choice in valid_range:
                return choice
            print(f"Please enter a number between {min(valid_range)} and {max(valid_range)}")
        except ValueError:
            print("Please enter a valid number")

def ensure_dataset_exists():
    """Ensure dataset exists locally or download it"""
    dataset_dir = "datasets"
    dataset_path = os.path.join(dataset_dir, "natural_reasoning")
    
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    if os.path.exists(dataset_path):
        print("Using locally cached dataset...")
        try:
            ds = load_dataset("json", data_files=f"{dataset_path}/train.json")
            return ds
        except Exception as e:
            print(f"Error loading cached dataset: {e}")
            print("Will attempt to download fresh copy...")
    
    print("Downloading dataset from Hugging Face...")
    try:
        ds = load_dataset("facebook/natural_reasoning")
        
        # Save dataset locally
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        
        # Save each split
        for split in ds.keys():
            split_file = os.path.join(dataset_path, f"{split}.json")
            ds[split].to_json(split_file)
        
        print(f"Dataset saved locally to {dataset_path}")
        return ds
    
    except Exception as e:
        raise Exception(f"Failed to download dataset: {e}")

# ...existing code...

def extract_valid_texts(ds):
    """Extract and validate texts from dataset"""
    texts = []
    print("Analyzing dataset structure...")
    
    # Print a sample item to debug
    sample_item = ds[0]
    print("\nSample item structure:")
    for key, value in sample_item.items():
        print(f"{key}: {type(value)} - Example: {str(value)[:100]}...")
    
    for item in tqdm(ds, desc="Extracting texts"):
        try:
            # Extract texts from relevant fields
            if 'premise' in item and item['premise']:
                text = str(item['premise']).strip()
                if len(text) >= 30 and len(text.split()) >= 5:
                    texts.append(text)
            
            if 'hypothesis' in item and item['hypothesis']:
                text = str(item['hypothesis']).strip()
                if len(text) >= 30 and len(text.split()) >= 5:
                    texts.append(text)
            
            # Only include high-quality explanations
            if 'explanation' in item and item['explanation']:
                text = str(item['explanation']).strip()
                if len(text) >= 50 and len(text.split()) >= 8:  # Higher threshold for explanations
                    texts.append(text)
            
            # Process any additional text fields
            for field in ['context', 'conclusion', 'question']:
                if field in item and item[field]:
                    text = str(item[field]).strip()
                    if len(text) >= 30 and len(text.split()) >= 5:
                        texts.append(text)
        
        except Exception as e:
            continue  # Skip problematic items
    
    # Remove duplicates while preserving order
    texts = list(dict.fromkeys(texts))
    
    # Additional validation
    valid_texts = []
    for text in texts:
        # Check for minimum word variety
        words = text.split()
        unique_words = set(words)
        if len(unique_words) >= len(words) * 0.4:  # At least 40% unique words
            valid_texts.append(text)
    
    return valid_texts

# Add worker initialization function at module level
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Update the main training section
def ensure_checkpoint_dir():
    """Ensure checkpoint directory exists, using Google Drive if in Colab"""
    if IN_COLAB:
        try:
            drive.mount('/content/drive')
            checkpoint_dir = "/content/drive/MyDrive/TITSR1_checkpoints"
        except Exception as e:
            print(f"Failed to mount Google Drive: {e}")
            checkpoint_dir = "checkpoint"  # Fallback to local
    else:
        checkpoint_dir = "checkpoint"
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return checkpoint_dir

def get_latest_checkpoint():
    """Get the path of the latest checkpoint from Google Drive if in Colab"""
    checkpoint_dir = ensure_checkpoint_dir()
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not checkpoints:
        return None
    
    # Sort checkpoints by epoch number
    checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return os.path.join(checkpoint_dir, checkpoints[-1])

def save_checkpoint(epoch, model, optimizer, scheduler, loss, tokenizer, config, is_best=False):
    """Save training checkpoint to Google Drive if in Colab"""
    checkpoint_dir = ensure_checkpoint_dir()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'config': config,
        'vocab_size': tokenizer.get_vocab_size(),
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model if needed
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
    
    # Keep only last 3 checkpoints to save space
    checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_')])
    for old_checkpoint in checkpoints[:-3]:
        os.remove(os.path.join(checkpoint_dir, old_checkpoint))
    
    # Also save tokenizer in the same directory
    tokenizer_path = os.path.join(checkpoint_dir, 'tokenizer.json')
    tokenizer.save(tokenizer_path)
    
    return checkpoint_path

def load_checkpoint(model, optimizer, scheduler, tokenizer):
    """Try to load the latest checkpoint from Google Drive if in Colab"""
    checkpoint_path = get_latest_checkpoint()
    if not checkpoint_path:
        return None, 0, float('inf')
    
    try:
        print(f"Found checkpoint at {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if it exists
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Try to load tokenizer from the same directory
        tokenizer_path = os.path.join(os.path.dirname(checkpoint_path), 'tokenizer.json')
        if os.path.exists(tokenizer_path):
            tokenizer.load(tokenizer_path)
        
        # Verify vocabulary size matches
        if checkpoint['vocab_size'] != tokenizer.get_vocab_size():
            print("Warning: Checkpoint vocabulary size doesn't match current tokenizer")
            return None, 0, float('inf')
        
        print(f"Successfully loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint['config'], checkpoint['epoch'], checkpoint['loss']
        
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        return None, 0, float('inf')

import json

# Add this function after the load_checkpoint function and before main()
def train_on_model_json(model, tokenizer, optimizer, device):
    """Train the model on model.json data first for self-awareness"""
    print_header("Training on Self-Identity Data")
    
    try:
        # Load model.json
        with open('/home/awion/TITSR1/datasets/model.json', 'r') as f:
            model_data = json.load(f)
        
        # Extract relevant texts for self-awareness
        identity_texts = []
        
        # Process model info
        info = model_data['model_info']
        identity_texts.append(f"I am {info['name']}, version {info['version']}. {info['description']}")
        
        # Process architecture
        arch = model_data['architecture']
        identity_texts.append(
            f"My architecture consists of a {arch['type']} model with {arch['parameters']} parameters, "
            f"{arch['layers']} layers, and {arch['attention_heads']} attention heads."
        )
        
        # Process capabilities
        for category, abilities in model_data['capabilities'].items():
            capability_text = f"In terms of {category}, I am capable of "
            ability_list = [f"{k.replace('_', ' ')} with {v:.0%} proficiency" 
                          for k, v in abilities.items()]
            capability_text += ", ".join(ability_list)
            identity_texts.append(capability_text)
        
        # Process datasets info
        for dataset in model_data['datasets']:
            if dataset['name'] == 'self_identity_corpus':
                for entry in dataset['sample_entries']:
                    identity_texts.extend([
                        entry['identity_statement'],
                        entry['self_description'],
                        entry['purpose_statement']
                    ])
        
        # Create a small dataset for identity training
        identity_dataset = NaturalReasoningDataset(identity_texts, tokenizer)
        identity_loader = DataLoader(
            identity_dataset,
            batch_size=4,  # Small batch size for careful learning
            shuffle=True,
            num_workers=1,
            pin_memory=True
        )
        
        # Train for a few epochs on identity data
        model.train()
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        print("\nTraining on self-identity data...")
        identity_epochs = 50  # More epochs on small identity dataset
        
        for epoch in range(identity_epochs):
            running_loss = 0.0
            pbar = tqdm(identity_loader, desc=f"Identity Training Epoch {epoch+1}/{identity_epochs}")
            
            for inputs, targets in pbar:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                with autocast():
                    outputs = model(inputs)
                    outputs = outputs.reshape(-1, tokenizer.get_vocab_size())
                    targets = targets.reshape(-1)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                running_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
            
            avg_loss = running_loss / len(identity_loader)
            print(f"Identity training epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            # Generate sample identity response every 10 epochs
            if (epoch + 1) % 10 == 0:
                model.eval()
                prompts = [
                    "I am",
                    "My capabilities include",
                    "My purpose is",
                    "As an AI model, I"
                ]
                print("\nCurrent identity understanding:")
                for prompt in prompts:
                    with torch.no_grad():
                        sample = generate_text(
                            model,
                            tokenizer,
                            seed_text=prompt,
                            max_length=100,
                            temperature=0.7
                        )
                        print(f"\nPrompt: '{prompt}'\nOutput: {sample}\n")
                model.train()
        
        print("\nSelf-identity training completed!")
        return True
        
    except Exception as e:
        print(f"Error during identity training: {str(e)}")
        return False

# Modify the main() function to include identity training
def main():
    # Optimized configuration for T4 GPU
    batch_size = 16  # Reduced batch size
    gradient_accumulation_steps = 4  # Effective batch size = 16 * 4 = 64
    seq_length = 48  # Reduced sequence length
    epochs = 30
    initial_lr = 5e-4
    vocab_size = 25000  # Reduced vocabulary size
    
    # Reduced model size
    embedding_dim = 256  # Reduced from 384
    hidden_dim = 512    # Reduced from 768
    num_layers = 3      # Reduced from 4
    dropout = 0.1

    # Enable mixed precision training with explicit device
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    print_header("TitanoSR1 Training Interface")

    # Initialize should_retrain
    should_retrain = False

    # Check if model exists and get training mode
    if check_model_exists():
        print("Model Status: Existing model found!\n")
        options = [
            "Use existing model",
            "Retrain existing model",
            "Train new model"
        ]
        print_menu(options)
        choice = get_choice("Enter your choice", range(1, 4))
        
        if choice == 1:
            print("\n→ Using existing model. Run generate.py to use the model.")
            return
        elif choice == 2:
            print("\n→ Loading existing model for retraining...")
            model, tokenizer = load_existing_model()
            should_retrain = True
        else:
            print("\n→ Training new model...")
            should_retrain = False
    else:
        print("Model Status: No existing model found.")
        print("\n→ Proceeding with new model training...")

    # Dataset loading
    print_header("Dataset Loading")
    try:
        # Import and login
        from datasets import load_dataset
        from huggingface_hub import login
        login(token=" ", write_permission=False)
        print("Successfully logged in to Hugging Face!")
        
        # Load or download dataset
        ds = ensure_dataset_exists()
        print("Dataset loaded successfully!")
        
        # Get training split
        train_ds = ds['train'] if isinstance(ds, dict) else ds
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        print("\nPlease check:")
        print("1. Internet connection")
        print("2. Install required packages:")
        print("   pip install --upgrade datasets huggingface_hub")
        return

    try:
        print_header("Training Process")
        
        # Extract and validate texts
        texts = extract_valid_texts(train_ds)
        
        if not texts:
            raise ValueError("No valid texts extracted! Dataset may be empty or malformed.")
        
        if not should_retrain:
            print("\nCreating new tokenizer and model...")
            tokenizer = SimpleTokenizer(vocab_size=vocab_size)
            tokenizer.train_from_texts(texts)
            
            # Verify vocabulary size
            actual_vocab_size = tokenizer.get_vocab_size()
            print(f"Created vocabulary with {actual_vocab_size} tokens")
            if actual_vocab_size <= 4:
                raise ValueError("Vocabulary creation failed - only special tokens were added")
            
            # Create new model
            model = TitanoSR1(
                vocab_size=actual_vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            ).to(device)
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Add identity training here
        print("\nStarting self-identity training...")
        identity_trained = train_on_model_json(model, tokenizer, optimizer, device)
        if not identity_trained:
            print("Warning: Self-identity training failed or was skipped.")
        
        # Create dataset and begin training
        dataset = NaturalReasoningDataset(texts, tokenizer, seq_length=seq_length)
        
        # DataLoader with optimized settings
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            worker_init_fn=worker_init_fn,  # Now using module-level function
            drop_last=True,
            persistent_workers=True
        )
        
        # Initialize criterion and optimizer with improved scheduler
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.AdamW(model.parameters(), lr=initial_lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        
        # Implement a proper learning rate scheduler
        # Warmup for 10% of training followed by cosine decay
        warmup_steps = int(0.1 * epochs * len(dataloader))
        total_steps = epochs * len(dataloader)
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # Cosine decay after warmup
            return 0.5 * (1.0 + math.cos(math.pi * (current_step - warmup_steps) / (total_steps - warmup_steps)))
            
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Create configuration dictionary
        config = {
            'batch_size': batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'seq_length': seq_length,
            'epochs': epochs,
            'initial_lr': initial_lr,
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout
        }
        
        # Try to load checkpoint
        loaded_config, start_epoch, best_loss = load_checkpoint(model, optimizer, scheduler, tokenizer)
        
        if loaded_config:
            # Update configuration if loaded successfully
            config = loaded_config
            print("Resuming training from checkpoint...")
        else:
            start_epoch = 0
            best_loss = float('inf')
            print("Starting training from scratch...")
        
        # Training loop with validation and early stopping
        print(f"Starting training from epoch {start_epoch + 1} to {epochs}")
        
        patience = 5  # Early stopping patience
        patience_counter = 0
        
        for epoch in range(start_epoch, epochs):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad()
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch_idx, (inputs, targets) in enumerate(pbar):
                # Move data to GPU
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                
                # Mixed precision forward pass
                with autocast():
                    outputs = model(inputs)
                    outputs = outputs.reshape(-1, tokenizer.get_vocab_size())
                    targets = targets.reshape(-1)
                    loss = criterion(outputs, targets)
                    loss = loss / gradient_accumulation_steps  # Normalize loss

                # Mixed precision backward pass
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    # Unscale gradients for clipping
                    scaler.unscale_(optimizer)
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Update weights with scaled gradients
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

                running_loss += loss.item() * gradient_accumulation_steps
                
                # Update progress bar less frequently
                if batch_idx % 10 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix(loss=loss.item() * gradient_accumulation_steps, lr=f"{current_lr:.6f}")

                # Generate samples less frequently
                if batch_idx % 1000 == 0 and batch_idx > 0:
                    model.eval()
                    with torch.no_grad(), autocast():
                        sample = generate_text(
                            model,
                            tokenizer,
                            seed_text="The reason for this conclusion is",
                            max_length=50  # Reduced length for faster generation
                        )
                    print(f"\nSample during training:\n{sample}\n")
                    model.train()
            
            avg_loss = running_loss / len(dataloader)
            print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            # Save checkpoint after each epoch
            is_best = avg_loss < best_loss
            checkpoint_path = save_checkpoint(
                epoch + 1,
                model,
                optimizer,
                scheduler,
                avg_loss,
                tokenizer,
                config,
                is_best
            )
            print(f"Checkpoint saved: {checkpoint_path}")
            
            # Early stopping check
            if is_best:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Generate sample text at the end of each epoch
            if (epoch + 1) % 1 == 0:
                # Generate with different prompts
                prompts = [
                    "The reason for this conclusion is",
                    "This statement is true because",
                    "The logical explanation is that",
                    "We can infer that"
                ]
                
                for prompt in prompts:
                    sample = generate_text(
                        model, 
                        tokenizer, 
                        seed_text=prompt,
                        temperature=0.8,
                        top_p=0.92
                    )
                    print(f"\nSample with prompt '{prompt}':\n{sample}\n")
            
            # Save checkpoints
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                checkpoint_path = f"titano_sr1_checkpoint_epoch_{epoch+1}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'vocab_size': tokenizer.get_vocab_size(),
                    'embedding_dim': embedding_dim,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'dropout': dropout,
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save the best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'vocab_size': tokenizer.get_vocab_size(),
                    'embedding_dim': embedding_dim,
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'dropout': dropout,
                }, "titano_sr1_best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Save the final model and tokenizer
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab_size': tokenizer.get_vocab_size(),
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout,
        }, "titano_sr1_model.pt")
        tokenizer.save("titano_sr1_tokenizer.json")
        print("Training completed! Model and tokenizer saved.")
        
        # Final generation examples
        print("\nFinal model output examples:")
        for prompt in [
            "The argument is valid because",
            "The flaw in this reasoning is",
            "Based on these premises, we can conclude that",
            "The evidence suggests that"
        ]:
            sample = generate_text(
                model, 
                tokenizer, 
                seed_text=prompt, 
                max_length=150,
                temperature=0.7,
                top_p=0.9
            )
            print(f"\nPrompt: '{prompt}'\nOutput: {sample}\n")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("\nThis might be due to authentication issues with Hugging Face.")
        print("Please make sure you've logged in with 'huggingface-cli login'")
        print("You may need to install additional packages:")
        print("pip install datasets huggingface_hub")

if __name__ == "__main__":
    # Enable memory efficient optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Set multiprocessing start method if not Windows
    if os.name != 'nt':  # Skip on Windows
        mp.set_start_method('spawn', force=True)
    
    main()
