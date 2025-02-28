import pandas as pd
import json
import time
import subprocess
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Set, Tuple

class DialogueProcessor:
    def __init__(self, root_path: str, similarity_threshold: float = 0.7):
        self.root_path = root_path
        self.similarity_threshold = similarity_threshold
        self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load KG entities
        with open(f'{root_path}/entity2id.json', 'r') as f:
            self.entity2id = json.load(f)
            
        # Setup entity embeddings
        self.setup_entity_embeddings()
        
    def setup_entity_embeddings(self):
        """Pre-compute embeddings for KG entities"""
        entity_names = list(self.entity2id.keys())
        self.entity_embeddings = self.sentence_encoder.encode(entity_names)
        self.entity_names = entity_names
        self.entity_embeddings = torch.tensor(self.entity_embeddings)
    
    def find_similar_entities(self, dialogue_entities: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        """Find similar KG entities for each dialogue entity"""
        matches = {}
        for d_ent in dialogue_entities:
            # First try exact match
            exact_match = next((kg_ent for kg_ent in self.entity_names 
                              if d_ent.lower() == kg_ent.lower()), None)
            if exact_match:
                matches[d_ent] = [(exact_match, 1.0)]
                continue
                
            # If no exact match, use semantic similarity
            d_emb = self.sentence_encoder.encode(d_ent)
            d_emb = torch.tensor(d_emb)
            
            similarities = F.cosine_similarity(
                d_emb.unsqueeze(0),
                self.entity_embeddings,
                dim=1
            )
            
            # Get top matches above threshold
            matched_indices = torch.where(similarities > self.similarity_threshold)[0]
            matched_scores = similarities[matched_indices]
            
            if len(matched_indices) > 0:
                matched_entities = [(self.entity_names[idx], score.item()) 
                                  for idx, score in zip(matched_indices, matched_scores)]
                matched_entities.sort(key=lambda x: x[1], reverse=True)
                matches[d_ent] = matched_entities[:5]  # Keep top 5 matches
            else:
                matches[d_ent] = []
                
        return matches
    
    def extract_subgraph(self, matched_entities: Set[str], df_relations: pd.DataFrame) -> pd.DataFrame:
        """Extract subgraph for matched entities"""
        # Filter relations where either source or target is in matched entities
        subgraph = df_relations[
            (df_relations['source'].isin(matched_entities)) | 
            (df_relations['target'].isin(matched_entities))
        ].copy()
        
        return subgraph
    
    def generate_text_description(self, subgraph: pd.DataFrame, entity_matches: Dict[str, List[Tuple[str, float]]]) -> str:
        """Generate text description from subgraph relationships"""
        descriptions = []
        
        # Sort entities by similarity score
        sorted_entities = sorted(
            [(ent, matches[0][1]) for ent, matches in entity_matches.items() if matches],
            key=lambda x: x[1],
            reverse=True
        )
        
        for entity, score in sorted_entities:
            # Get all relationships involving this entity
            entity_rels = subgraph[
                (subgraph['source'] == entity) | (subgraph['target'] == entity)
            ]
            
            if not entity_rels.empty:
                entity_desc = [f"Entity '{entity}' (similarity: {score:.2f}):"]
                for _, rel in entity_rels.iterrows():
                    entity_desc.append(f"- {rel['description']}")
                descriptions.append("\n".join(entity_desc))
        
        return "\n\n".join(descriptions)
    
    def process_dialogue(self, dialogue: str) -> str:
        """Process a single dialogue record"""
        # Write dialogue to current_line.txt
        with open(f'{self.root_path}/input/current_line.txt', 'w') as f:
            f.write(dialogue)
            
        # Run GraphRAG indexing
        process = subprocess.run(
            ['graphrag', 'index', '--root', self.root_path],
            capture_output=True,
            text=True
        )
        
        if process.returncode != 0:
            raise Exception(f"GraphRAG indexing failed: {process.stderr}")
            
        # Read GraphRAG output
        df_relations = pd.read_parquet(f'{self.root_path}/output/create_final_relationships.parquet')
        dialogue_entities = list(set(df_relations['source'].unique()) | set(df_relations['target'].unique()))
        
        # Find similar entities in original KG
        entity_matches = self.find_similar_entities(dialogue_entities)
        
        # Extract subgraph for matched entities
        matched_entities = {match[0] for matches in entity_matches.values() 
                          for match in matches if match[1] > self.similarity_threshold}
        subgraph = self.extract_subgraph(matched_entities, df_relations)
        
        # Generate text description
        text_description = self.generate_text_description(subgraph, entity_matches)
        
        return text_description

def process_all_dialogues(root_path: str):
    processor = DialogueProcessor(root_path)
    
    with open(f'{root_path}/dialogues/test_data.txt', 'r') as file:
        for i, line in enumerate(file):
            print(f"\nProcessing dialogue {i}")
            dialogue = line.strip()
            
            try:
                text_description = processor.process_dialogue(dialogue)
                print(f"\nGenerated description:\n{text_description}")
                
                # Here you would feed this into your recommendation model
                # recommendation_prompt = f"{dialogue} [Context] {text_description}"
                
            except Exception as e:
                print(f"Error processing dialogue {i}: {e}")
            
            time.sleep(2)

if __name__ == "__main__":
    ROOT_PATH = '/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG'
    process_all_dialogues(ROOT_PATH)