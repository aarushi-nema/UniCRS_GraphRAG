import pandas as pd
import json
import time
import subprocess
import torch
from model_prompt import KGPrompt
from pre_rec_dataset_dbpedia import DBpedia 

# First load the entities dataframe that has all the original entity and relationship info
kg = DBpedia(dataset='redial', debug=False).get_entity_kg_info()

# Initialize the RGCN-based prompt encoder
prompt_encoder = KGPrompt(
    hidden_size=768,  # Assuming default hidden size
    token_hidden_size=768,
    n_head=12,
    n_layer=12,
    n_block=2,
    n_entity=kg['num_entities'],
    num_relations=kg['num_relations'],
    num_bases=8,
    edge_index=kg['edge_index'],
    edge_type=kg['edge_type'],
)

with open('/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG/dialogues/test_data.txt', 'r') as file:
    for i,line in enumerate(file):
        print("Processing line ", i)
        line = line.strip()
        with open('/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG/input/current_line.txt', 'w') as current_file:
           current_file.write(line)

        # Run the graphrag command and wait for it to complete
        process = subprocess.run(['graphrag', 'index', '--root', '/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG/'], 
                              capture_output=True,  # Capture the output
                              text=True)  # Convert output to string
        
        if process.returncode == 0:
            print("Command completed successfully")
            
            # Read the generated parquet files
            entities_df = pd.read_parquet('/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG/output/create_final_entities.parquet')
            relations_df = pd.read_parquet('/home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG/output/create_final_relationships.parquet')
            
            print("Found entities:", len(entities_df))
            print("First 5 entities", entities_df['title'])
            print("Found relationships:", len(relations_df))

            # Create a mapping from entity id to human_readable_id
            entity_id_map = dict(zip(entities_df['id'], entities_df['human_readable_id']))

            # Create edge_index and edge_type tensors from the relations DataFrame
            edge_list = []
            edge_types = []
            
            for _, row in relations_df.iterrows():
                try:
                    # Get source and target IDs using the mapping
                    source_entity = entity_id_map[row['source']]
                    target_entity = entity_id_map[row['target']]
                    
                    source_id = int(source_entity)
                    target_id = int(target_entity)
                    
                    # For now use a single relation type (0)
                    relation_id = 0
                    edge_list.append([source_id, target_id])
                    edge_types.append(relation_id)
                except Exception as e:
                    print(f"Error processing row: {row}")
                    print(f"Error: {e}")
                    continue

            if edge_list:  # Only proceed if we have edges
                # Convert to tensors
                dialogue_edge_index = torch.tensor(edge_list, dtype=torch.long).t()
                dialogue_edge_type = torch.tensor(edge_types, dtype=torch.long)

                print("\nCreated tensors:")
                print("Edge index shape:", dialogue_edge_index.shape)
                print("Edge type shape:", dialogue_edge_type.shape)

                # Store original graph info
                orig_edge_index = prompt_encoder.edge_index
                orig_edge_type = prompt_encoder.edge_type

                # Temporarily replace with dialogue subgraph
                prompt_encoder.edge_index.data = dialogue_edge_index
                prompt_encoder.edge_type.data = dialogue_edge_type

                # Get embeddings using the RGCN
                with torch.no_grad():
                    dialogue_embeddings = prompt_encoder.get_entity_embeds()
                    print("Got dialogue embeddings shape:", dialogue_embeddings.shape)

                # Restore original graph
                prompt_encoder.edge_index.data = orig_edge_index
                prompt_encoder.edge_type.data = orig_edge_type
            else:
                print("No valid edges found in this dialogue")
        else:
            print("Command failed with return code:", process.returncode)
        
        time.sleep(2)