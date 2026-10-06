"""
Weighted Knowledge Graph Builder for Text-Video Retrieval
==========================================================
Author: Senior Python Data Engineer & NLP Specialist
Purpose: Transform triplet text files into structured weighted knowledge graphs

This script processes cleaned entity files (msrvtt/msvd_cleaned_entities.txt)
and builds a hierarchical JSON knowledge graph with:
- Lemmatization & POS tagging
- Conditional probability weights
- Strong relationship filtering
"""

import spacy
import json
from collections import defaultdict, Counter
from tqdm import tqdm
import os
from typing import Dict, List, Tuple, Optional


class WeightedKGBuilder:
    """
    Builds a weighted knowledge graph from triplet text files.
    
    Attributes:
        nlp: Spacy NLP model for lemmatization and POS tagging
        head_global_count: Counter for global occurrences of each head
        triplet_count: Counter for (head, tail, relation) triplets
        head_pos_tags: Counter for POS tags of each head
        tail_pos_tags: Counter for POS tags of each tail
    """
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize the KG builder with specified spacy model.
        
        Args:
            spacy_model: Name of the spacy model to load
        """
        print(f"[INFO] Loading spacy model: {spacy_model}")
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            print(f"[ERROR] Model '{spacy_model}' not found. Installing...")
            os.system(f"python -m spacy download {spacy_model}")
            self.nlp = spacy.load(spacy_model)
        
        # Statistical counters
        self.head_global_count = Counter()  # Count(head_lemma)
        self.triplet_count = Counter()      # Count(head_lemma, tail_lemma, relation_lemma)
        self.head_pos_tags = Counter()      # POS tags for heads
        self.tail_pos_tags = Counter()      # POS tags for tails
        
        print("[INFO] KG Builder initialized successfully")
    
    def clean_entity(self, text: str, entity_type: str = "noun") -> Tuple[str, str]:
        """
        Clean and lemmatize an entity (head/tail) or relation.
        
        Args:
            text: Raw text to process
            entity_type: Type of entity - "noun" for head/tail, "verb" for relation
            
        Returns:
            Tuple of (lemmatized_text, POS_tag)
            
        Logic:
            - For nouns (head/tail): Keep compound nouns, remove determiners
            - For verbs (relation): Remove auxiliaries, keep main verb
        """
        doc = self.nlp(text.strip())
        
        if entity_type == "noun":
            # Keep compound nouns, remove determiners (a, an, the)
            tokens = [
                token.lemma_.lower() 
                for token in doc 
                if token.pos_ not in ["DET"] and not token.is_space
            ]
            lemma = " ".join(tokens) if tokens else text.strip().lower()
            
            # Get majority POS tag (usually NOUN or PROPN)
            pos_tags = [token.pos_ for token in doc if token.pos_ not in ["DET", "SPACE"]]
            pos = pos_tags[0] if pos_tags else "NOUN"
            
        else:  # entity_type == "verb"
            # Remove auxiliary verbs (is, are, be, have, etc.)
            tokens = [
                token.lemma_.lower() 
                for token in doc 
                if token.pos_ not in ["AUX", "DET"] and not token.is_space
            ]
            lemma = " ".join(tokens) if tokens else text.strip().lower()
            
            # Get POS tag (usually VERB)
            pos_tags = [token.pos_ for token in doc if token.pos_ not in ["AUX", "DET", "SPACE"]]
            pos = pos_tags[0] if pos_tags else "VERB"
        
        return lemma, pos
    
    def parse_line(self, line: str) -> Optional[Tuple[str, str, str, str, str]]:
        """
        Parse a single line from the input file.
        
        Args:
            line: Input line in format "head &tail &relation"
            
        Returns:
            Tuple of (head_lemma, head_pos, tail_lemma, tail_pos, relation_lemma)
            or None if line is invalid
        """
        parts = line.strip().split(" &")
        
        # Validate format: must have exactly 3 parts
        if len(parts) != 3:
            return None
        
        head_raw, tail_raw, relation_raw = parts
        
        # Clean and lemmatize each component
        head_lemma, head_pos = self.clean_entity(head_raw, "noun")
        tail_lemma, tail_pos = self.clean_entity(tail_raw, "noun")
        relation_lemma, _ = self.clean_entity(relation_raw, "verb")
        
        return head_lemma, head_pos, tail_lemma, tail_pos, relation_lemma
    
    def load_and_count(self, file_path: str) -> None:
        """
        Load data from file and compute statistical counts.
        
        Args:
            file_path: Path to the input text file
            
        Process:
            1. Read file line by line
            2. Parse and lemmatize triplets
            3. Update global counters
        """
        print(f"[INFO] Loading data from: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Count total lines for progress bar
        with open(file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        print(f"[INFO] Processing {total_lines:,} lines...")
        
        valid_count = 0
        invalid_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="Parsing triplets"):
                # Skip empty lines
                if not line.strip():
                    invalid_count += 1
                    continue
                
                # Parse line
                result = self.parse_line(line)
                
                if result is None:
                    invalid_count += 1
                    continue
                
                head_lemma, head_pos, tail_lemma, tail_pos, relation_lemma = result
                
                # Update counters
                self.head_global_count[head_lemma] += 1
                self.triplet_count[(head_lemma, tail_lemma, relation_lemma)] += 1
                self.head_pos_tags[(head_lemma, head_pos)] += 1
                self.tail_pos_tags[(tail_lemma, tail_pos)] += 1
                
                valid_count += 1
        
        print(f"[INFO] Parsing complete:")
        print(f"  ✓ Valid triplets: {valid_count:,}")
        print(f"  ✗ Invalid/empty lines: {invalid_count:,}")
        print(f"  • Unique heads: {len(self.head_global_count):,}")
        print(f"  • Unique triplets: {len(self.triplet_count):,}")
    
    def get_majority_pos(self, entity: str, pos_counter: Counter) -> str:
        """
        Get the most common POS tag for an entity.
        
        Args:
            entity: The entity (head or tail)
            pos_counter: Counter of (entity, pos) tuples
            
        Returns:
            Most frequent POS tag for this entity
        """
        # Find all POS tags for this entity
        entity_pos_tags = [
            pos for (ent, pos) in pos_counter.keys() if ent == entity
        ]
        
        if not entity_pos_tags:
            return "NOUN"  # Default
        
        # Count occurrences of each POS tag
        pos_counts = Counter()
        for (ent, pos) in pos_counter.keys():
            if ent == entity:
                pos_counts[pos] += pos_counter[(ent, pos)]
        
        return pos_counts.most_common(1)[0][0]
    
    def calculate_weights(self, threshold_weight: float = 0.05, threshold_count: int = 1) -> Dict:
        """
        Build the knowledge graph with conditional probability weights.
        
        Args:
            threshold_weight: Minimum weight for "strong" relationships
            threshold_count: Minimum count for "strong" relationships
            
        Returns:
            Dictionary representing the weighted knowledge graph
            
        Mathematical Formula:
            weight = Count(Head, Tail, Relation) / Global_Count(Head)
            
            This represents the conditional probability:
            P(Tail, Relation | Head)
            
            Interpretation: Given we see "Head", what is the probability
            it's connected to "Tail" via "Relation"?
        """
        print("[INFO] Calculating weights and building KG structure...")
        
        kg_data = {}
        
        # Group triplets by head
        head_to_triplets = defaultdict(list)
        for (head, tail, relation), count in self.triplet_count.items():
            head_to_triplets[head].append((tail, relation, count))
        
        # Build KG for each head
        for head in tqdm(self.head_global_count.keys(), desc="Building KG"):
            # Get global count for this head
            global_count = self.head_global_count[head]
            
            # Get majority POS tag
            head_pos = self.get_majority_pos(head, self.head_pos_tags)
            
            # Build neighbors list
            neighbors = []
            for tail, relation, triplet_count in head_to_triplets[head]:
                # Calculate conditional probability weight
                weight = triplet_count / global_count
                
                # Get tail POS
                tail_pos = self.get_majority_pos(tail, self.tail_pos_tags)
                
                # Determine if this is a "strong" relationship
                is_strong = (weight > threshold_weight) and (triplet_count > threshold_count)
                
                neighbors.append({
                    "tail": tail,
                    "tail_pos": tail_pos,
                    "relation": relation,
                    "weight": round(weight, 4),  # Round to 4 decimal places
                    "count": triplet_count,
                    "is_strong": is_strong
                })
            
            # Sort neighbors by weight (descending)
            neighbors.sort(key=lambda x: x["weight"], reverse=True)
            
            # Add to KG
            kg_data[head] = {
                "pos": head_pos,
                "global_count": global_count,
                "neighbors": neighbors
            }
        
        print(f"[INFO] KG construction complete!")
        print(f"  • Total nodes (heads): {len(kg_data):,}")
        
        # Calculate statistics
        total_edges = sum(len(node["neighbors"]) for node in kg_data.values())
        strong_edges = sum(
            sum(1 for neighbor in node["neighbors"] if neighbor["is_strong"])
            for node in kg_data.values()
        )
        
        print(f"  • Total edges: {total_edges:,}")
        print(f"  • Strong edges: {strong_edges:,} ({strong_edges/total_edges*100:.1f}%)")
        
        return kg_data
    
    def save_kg(self, kg_data: Dict, output_path: str) -> None:
        """
        Save the knowledge graph to a JSON file.
        
        Args:
            kg_data: The knowledge graph dictionary
            output_path: Path to save the JSON file
        """
        print(f"[INFO] Saving KG to: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(kg_data, f, indent=4, ensure_ascii=False)
        
        # Get file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[SUCCESS] KG saved successfully! (Size: {file_size_mb:.2f} MB)")
    
    def build(self, input_path: str, output_path: str, 
              threshold_weight: float = 0.05, threshold_count: int = 1) -> None:
        """
        Complete pipeline to build weighted KG from text file.
        
        Args:
            input_path: Path to input text file
            output_path: Path to output JSON file
            threshold_weight: Minimum weight for strong relationships
            threshold_count: Minimum count for strong relationships
        """
        print("="*70)
        print("WEIGHTED KNOWLEDGE GRAPH BUILDER")
        print("="*70)
        
        # Step 1: Load and count
        self.load_and_count(input_path)
        
        # Step 2: Calculate weights
        kg_data = self.calculate_weights(threshold_weight, threshold_count)
        
        # Step 3: Save to JSON
        self.save_kg(kg_data, output_path)
        
        print("="*70)
        print("PROCESS COMPLETED SUCCESSFULLY!")
        print("="*70)


def main():
    """
    Main execution function.
    
    Processes both MSRVTT and MSVD datasets.
    """
    # Configuration
    SPACY_MODEL = "en_core_web_sm"
    THRESHOLD_WEIGHT = 0.05  # Minimum weight for strong edges (5% probability)
    THRESHOLD_COUNT = 1      # Minimum count for strong edges (must appear more than once)
    
    # Get script directory to save outputs in the same folder
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Dataset paths
    datasets = [
        {
            "name": "MSRVTT",
            "input": r"E:\Code_KLTN_E\0_CLIPKG4Clip\KG\msrvtt_cleaned_entities.txt",
            "output": os.path.join(SCRIPT_DIR, "msrvtt_weighted_kg.json")
        },
        {
            "name": "MSVD",
            "input": r"E:\Code_KLTN_E\0_CLIPKG4Clip\KG\msvd_cleaned_entities.txt",
            "output": os.path.join(SCRIPT_DIR, "msvd_weighted_kg.json")
        }
    ]
    
    # Process each dataset
    for dataset in datasets:
        print(f"\n{'='*70}")
        print(f"Processing {dataset['name']} Dataset")
        print(f"{'='*70}\n")
        
        # Check if input file exists
        if not os.path.exists(dataset["input"]):
            print(f"[WARNING] File not found: {dataset['input']}")
            print(f"[WARNING] Skipping {dataset['name']} dataset\n")
            continue
        
        # Build KG
        builder = WeightedKGBuilder(spacy_model=SPACY_MODEL)
        builder.build(
            input_path=dataset["input"],
            output_path=dataset["output"],
            threshold_weight=THRESHOLD_WEIGHT,
            threshold_count=THRESHOLD_COUNT
        )
        
        print("\n")


if __name__ == "__main__":
    main()
