import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys, os
from datetime import datetime
from typing import Dict, Any, List, Tuple
import json
from dataclasses import dataclass

@dataclass
class GenerationConfig:
    temperature: float = 1.0
    top_k: int = 40
    top_p: float = 0.9
    min_length: int = 50
    max_length: int = 1000
    repetition_penalty: float = 1.2

@dataclass
class PromptAnalysis:
    complexity: float
    tone: str
    theme: str
    literary_devices: List[str]
    suggested_temperature: float

class GenerationSystem:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.generation_history: List[Dict[str, Any]] = []
        self.generation_stats: Dict[str, float] = {
            "total_generations": 0,
            "avg_length": 0,
            "avg_temperature": 0
        }
        self.smart_defaults = {
            "narrative": {"temp": 0.7, "length": 300},
            "dialogue": {"temp": 0.8, "length": 200},
            "poetry": {"temp": 0.6, "length": 150},
            "monologue": {"temp": 0.75, "length": 250}
        }
        self.context_memory = []
    
    def log_generation(self, prompt: str, response: str, temperature: float, context: Dict[str, Any]) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "temperature": temperature,
            "length": len(response),
            "context": context
        }
        self.generation_history.append(entry)
        self._update_stats(entry)
    
    def _update_stats(self, entry: Dict[str, Any]) -> None:
        self.generation_stats["total_generations"] += 1
        n = self.generation_stats["total_generations"]
        self.generation_stats["avg_length"] = (
            (self.generation_stats["avg_length"] * (n-1) + entry["length"]) / n
        )
        self.generation_stats["avg_temperature"] = (
            (self.generation_stats["avg_temperature"] * (n-1) + entry["temperature"]) / n
        )
    
    def get_generation_insights(self) -> str:
        return (
            f"Generation Statistics:\n"
            f"Total Generations: {self.generation_stats['total_generations']}\n"
            f"Average Length: {self.generation_stats['avg_length']:.2f}\n"
            f"Average Temperature: {self.generation_stats['avg_temperature']:.2f}"
        )
    
    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze prompt to determine optimal generation parameters."""
        # Check for dialogue indicators
        if any(x in prompt.lower() for x in ["said", "spoke", "asked", "\"", "'"]):
            style = "dialogue"
        # Check for poetic indicators
        elif any(x in prompt.lower() for x in ["verse", "sonnet", "poem"]):
            style = "poetry"
        # Check for monologue indicators
        elif any(x in prompt.lower() for x in ["thought", "pondered", "mused"]):
            style = "monologue"
        # Default to narrative
        else:
            style = "narrative"
        
        return {
            "style": style,
            "temperature": self.smart_defaults[style]["temp"],
            "length": self.smart_defaults[style]["length"]
        }
    
    def post_process_text(self, text: str) -> str:
        """Clean up and format generated text."""
        # Fix incomplete sentences
        if not text.rstrip().endswith((".", "!", "?", "\"")):
            last_sentence = text.rstrip().rsplit(".", 1)[0] + "."
            text = last_sentence
        
        # Fix quotation marks
        quote_count = text.count("\"")
        if quote_count % 2 == 1:
            if text.rfind("\"") < len(text) - 20:  # If the last quote is not near the end
                text = text + "\""
            else:
                text = text[:text.rfind("\"")]
        
        # Ensure proper capitalization
        sentences = text.split(". ")
        sentences = [s.capitalize() for s in sentences]
        text = ". ".join(sentences)
        
        return text.strip()

class TextGenerator:
    def __init__(self, model, char_to_int: Dict[str, int], int_to_char: Dict[int, str]):
        self.model = model
        self.char_to_int = char_to_int
        self.int_to_char = int_to_char
        self.seq_length = 100  # Should match model's input length

    def analyze_prompt(self, prompt: str) -> PromptAnalysis:
        """Analyze prompt to determine optimal generation parameters."""
        # Calculate complexity based on unique words and structure
        words = prompt.split()
        complexity = len(set(words)) / len(words)
        
        # Simple tone analysis
        tones = {
            'formal': sum(1 for w in words if len(w) > 6),
            'poetic': prompt.count(',') + prompt.count(';'),
            'dramatic': prompt.count('!') + prompt.count('?')
        }
        tone = max(tones.items(), key=lambda x: x[1])[0]
        
        # Theme detection (simplified)
        themes = {
            'love': ['love', 'heart', 'passion'],
            'conflict': ['battle', 'fight', 'war'],
            'nature': ['tree', 'flower', 'sky']
        }
        theme_scores = {
            t: sum(1 for w in words if w.lower() in keywords)
            for t, keywords in themes.items()
        }
        theme = max(theme_scores.items(), key=lambda x: x[1])[0]
        
        # Detect literary devices (simplified)
        literary_devices = []
        if len(prompt) > 0:
            if any(w.endswith(('ly', 'ing')) for w in words):
                literary_devices.append('imagery')
            if len(set(w[0].lower() for w in words)) < len(words) * 0.7:
                literary_devices.append('alliteration')
        
        # Calculate suggested temperature based on analysis
        suggested_temp = min(1.0, 0.5 + complexity)
        
        return PromptAnalysis(
            complexity=complexity,
            tone=tone,
            theme=theme,
            literary_devices=literary_devices,
            suggested_temperature=suggested_temp
        )

    def _adjust_temperature(self, 
                          base_temp: float, 
                          generated_length: int, 
                          repetition_count: int) -> float:
        """Dynamically adjust temperature based on generation state."""
        # Increase temperature if we detect repetition
        rep_factor = 1.0 + (repetition_count * 0.1)
        
        # Gradually decrease temperature as we generate more text
        length_factor = max(0.8, 1.0 - (generated_length / 1000) * 0.2)
        
        return min(1.5, base_temp * rep_factor * length_factor)

    def _calculate_repetition_penalty(self, 
                                   generated_text: str, 
                                   window_size: int = 50) -> float:
        """Calculate repetition penalty based on recent output."""
        if len(generated_text) < window_size:
            return 1.0
            
        recent = generated_text[-window_size:]
        unique_chars = len(set(recent))
        penalty = 1.0 + (1.0 - unique_chars / window_size) * 0.5
        return min(2.0, penalty)

    def sample_with_dynamic_temp(self, 
                               logits: tf.Tensor, 
                               temperature: float, 
                               top_k: int, 
                               top_p: float) -> int:
        """Sample from logits with dynamic temperature and top-k/top-p filtering."""
        logits = logits / temperature
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < tf.math.top_k(logits, top_k)[0][..., -1, None]
            logits = tf.where(indices_to_remove, tf.fill(logits.shape, -float('inf')), logits)
        
        # Top-p filtering (nucleus sampling)
        if 0.0 < top_p < 1.0:
            sorted_logits = tf.sort(logits, direction='DESCENDING')
            cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits), axis=-1)
            mask = cumulative_probs > top_p
            mask = tf.concat([[False], mask[:-1]], axis=0)
            sorted_logits = tf.where(mask, -float('inf'), sorted_logits)
            logits = tf.gather(sorted_logits, tf.argsort(tf.argsort(logits)))
        
        # Sample from the filtered distribution
        probas = tf.nn.softmax(logits)
        return tf.random.categorical(tf.math.log(probas)[None, :], 1)[0, 0].numpy()

    def generate_text(self, 
                     prompt: str, 
                     config: Optional[GenerationConfig] = None) -> str:
        """Generate text with dynamic parameters and sophisticated sampling."""
        if config is None:
            config = GenerationConfig()
            
        # Analyze prompt and adjust parameters
        analysis = self.analyze_prompt(prompt)
        temperature = analysis.suggested_temperature
        
        # Initialize generation
        current_text = prompt
        generated_text = ""
        repetition_count = 0
        
        while len(generated_text) < config.max_length:
            # Prepare input sequence
            x_input = [self.char_to_int.get(char, self.char_to_int['<UNK>']) 
                      for char in current_text[-self.seq_length:]]
            x_input = tf.keras.preprocessing.sequence.pad_sequences(
                [x_input], maxlen=self.seq_length, padding='pre'
            )
            
            # Generate next character
            logits = self.model.predict(x_input, verbose=0)[0]
            
            # Adjust temperature based on generation state
            current_temp = self._adjust_temperature(
                temperature, 
                len(generated_text),
                repetition_count
            )
            
            # Apply repetition penalty
            rep_penalty = self._calculate_repetition_penalty(generated_text)
            
            # Sample next character
            next_char_index = self.sample_with_dynamic_temp(
                logits,
                current_temp * rep_penalty,
                config.top_k,
                config.top_p
            )
            
            next_char = self.int_to_char[next_char_index]
            generated_text += next_char
            current_text += next_char
            
            # Check for repetition
            if len(generated_text) > 10:
                last_10 = generated_text[-10:]
                if len(set(last_10)) < 5:
                    repetition_count += 1
                else:
                    repetition_count = max(0, repetition_count - 1)
            
            # Early stopping if needed
            if len(generated_text) >= config.min_length and next_char in '.!?':
                break
        
        return generated_text

def format_output(text: str, width: int = 80) -> str:
    """Format the generated text for better readability."""
    lines = []
    words = text.split()
    current_line = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 <= width:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)
    
    if current_line:
        lines.append(" ".join(current_line))
    
    return "\n".join(lines)

def main():
    # Initialize generation system
    gen_system = GenerationSystem("Titano-SR1")
    
    # Load model and data
    model_path = "Titano-SR1_best.keras" if os.path.exists("Titano-SR1_best.keras") else "Titano-SR1.h5"
    try:
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print("Error loading model:", e)
        sys.exit(1)
    
    try:
        text = open("shakespeare.txt", 'r', encoding='utf-8').read()
        char_to_int, int_to_char = {c: i for i, c in enumerate(sorted(set(text)))}, {i: c for i, c in enumerate(sorted(set(text)))}
    except Exception as e:
        print("Error loading dataset:", e)
        sys.exit(1)
    
    print("\nTitano-SR1 Text Generation Interface")
    print("=" * 50)
    print("Commands:")
    print("  temp=X.X     Set temperature (0.1-2.0)")
    print("  length=XXX   Set generation length")
    print("  stats        Show generation statistics")
    print("  exit         Quit the program")
    print("=" * 50)
    
    temperature = 0.8
    gen_length = 200
    
    while True:
        try:
            user_input = input("\nPrompt> ").strip()
            
            # Handle commands
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'stats':
                print("\n" + gen_system.get_generation_insights())
                continue
            elif user_input.startswith('temp='):
                try:
                    temperature = float(user_input.split('=')[1])
                    temperature = max(0.1, min(2.0, temperature))
                    print(f"Temperature set to {temperature}")
                except ValueError:
                    print("Invalid temperature value")
                continue
            elif user_input.startswith('length='):
                try:
                    gen_length = int(user_input.split('=')[1])
                    gen_length = max(50, min(1000, gen_length))
                    print(f"Generation length set to {gen_length}")
                except ValueError:
                    print("Invalid length value")
                continue
            
            # Get smart parameters based on prompt
            params = gen_system.analyze_prompt(user_input)
            temperature = params["temperature"]
            gen_length = params["length"]
            
            print(f"\nGenerating {params['style']} response...")
            generator = TextGenerator(model, char_to_int, int_to_char)
            config = GenerationConfig(
                temperature=temperature,
                top_k=40,
                top_p=0.9,
                min_length=100,
                max_length=gen_length
            )
            response = generator.generate_text(user_input, config)
            
            # Post-process the generated text
            response = gen_system.post_process_text(response)
            
            # Log generation
            gen_system.log_generation(
                user_input, response,
                temperature=temperature,
                context={"length": gen_length}
            )
            
            # Format and display output
            print("\nGenerated Text:")
            print("-" * 50)
            print(format_output(response))
            print("-" * 50)
            print(f"[Temperature: {temperature:.2f}, Length: {gen_length}]")
            
        except KeyboardInterrupt:
            print("\nGeneration interrupted by user.")
        except Exception as e:
            print(f"Error during generation: {e}")

if __name__ == "__main__":
    main()
