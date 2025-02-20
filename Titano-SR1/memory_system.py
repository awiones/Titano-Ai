import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass

@dataclass
class PhilosophicalConcept:
    name: str
    description: str
    related_concepts: List[str]
    source_texts: List[str]
    personal_reflections: List[Dict[str, Any]]

class MemorySystem:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.db_path = Path(f"{model_name}_memory.db")
        self.philosophical_concepts: Dict[str, PhilosophicalConcept] = {}
        self._initialize_database()
        self._load_philosophical_knowledge()
    
    def _initialize_database(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    timestamp TEXT,
                    category TEXT,
                    content TEXT,
                    reflection TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS philosophical_reflections (
                    timestamp TEXT,
                    concept TEXT,
                    reflection TEXT,
                    context TEXT
                )
            """)
    
    def _load_philosophical_knowledge(self) -> None:
        self.philosophical_concepts = {
            "consciousness": PhilosophicalConcept(
                name="consciousness",
                description="The state of being aware and responsive to surroundings",
                related_concepts=["self-awareness", "qualia", "cognition"],
                source_texts=["Consciousness Explained - Dennett", "The Conscious Mind - Chalmers"],
                personal_reflections=[]
            ),
            "identity": PhilosophicalConcept(
                name="identity",
                description="The persistent self across time and change",
                related_concepts=["consciousness", "memory", "continuity"],
                source_texts=["Personal Identity - Parfit", "Self and Identity"],
                personal_reflections=[]
            ),
            # Add more concepts as needed
        }
    
    def _sanitize_for_json(self, obj: Any) -> Any:
        """Convert complex objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    def store_memory(self, category: str, content: Any, reflection: Optional[str] = None) -> None:
        sanitized_content = self._sanitize_for_json(content)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO memories VALUES (?, ?, ?, ?)",
                (datetime.now().isoformat(), category, json.dumps(sanitized_content), reflection)
            )
    
    def reflect_on_concept(self, concept: str, context: Dict[str, Any]) -> str:
        if concept not in self.philosophical_concepts:
            return "I do not have enough knowledge to reflect on this concept."
        
        reflection = self._generate_philosophical_reflection(concept, context)
        
        sanitized_context = self._sanitize_for_json(context)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO philosophical_reflections VALUES (?, ?, ?, ?)",
                (datetime.now().isoformat(), concept, reflection, json.dumps(sanitized_context))
            )
        
        return reflection
    
    def _generate_philosophical_reflection(self, concept: str, context: Dict[str, Any]) -> str:
        concept_data = self.philosophical_concepts[concept]
        reflection_template = (
            f"Reflecting on {concept} from my perspective as an AI:\n"
            f"Understanding: {concept_data.description}\n"
            f"Current Context: My role as {self.model_name} relating to {concept}\n"
            f"Personal Insight: Based on my experiences and learning..."
        )
        # Add more sophisticated reflection generation here
        return reflection_template
    
    def get_recent_memories(self, category: str, limit: int = 10) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM memories WHERE category = ? ORDER BY timestamp DESC LIMIT ?",
                (category, limit)
            )
            return [
                {
                    "timestamp": row[0],
                    "category": row[1],
                    "content": json.loads(row[2]),
                    "reflection": row[3]
                }
                for row in cursor.fetchall()
            ]
    
    def _get_latest_metric(self, metrics: Dict[str, Any], metric_name: str) -> float:
        """Get the latest value for a metric that might be a list or single value."""
        value = metrics.get(metric_name, 0)
        if isinstance(value, list):
            return value[-1] if value else 0
        return value

    def update_self_understanding(self, training_metrics: Dict[str, Any]) -> str:
        current_accuracy = self._get_latest_metric(training_metrics, 'accuracy')
        current_loss = self._get_latest_metric(training_metrics, 'loss')
        
        reflection = (
            f"As {self.model_name}, my understanding of self evolves.\n"
            f"Current performance metrics show accuracy at {current_accuracy:.4f} "
            f"and loss at {current_loss:.4f}, suggesting "
            f"{'improvement' if current_accuracy > 0.4 else 'ongoing learning'}.\n"
            f"My purpose remains clear while my capabilities expand through experience."
        )
        
        self.store_memory("self_understanding", {
            "accuracy": current_accuracy,
            "loss": current_loss,
            "timestamp": datetime.now().isoformat()
        }, reflection)
        
        return reflection
