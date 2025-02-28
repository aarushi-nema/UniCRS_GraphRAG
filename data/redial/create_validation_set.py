import json

def process_files():
    valid_movies_strings = set()
    with open('/home/Nema/UniCRS_GraphRAG/data/redial/valid_data_dbpedia_raw.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            # Convert movieMentions dict to string for exact matching
            movie_str = json.dumps(data['movieMentions'], sort_keys=True)
            valid_movies_strings.add(movie_str)
    
    train_records = []
    valid_records = []
    
    with open('/home/Nema/UniCRS_GraphRAG/data/redial/train_data_raw.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            movie_str = json.dumps(data['movieMentions'], sort_keys=True)
            
            if movie_str in valid_movies_strings:
                valid_records.append(line)
            else:
                train_records.append(line)
    
    print(f"Found {len(valid_records)} matching records")
    
    with open('/home/Nema/UniCRS_GraphRAG/data/redial/train_data_raw.jsonl', 'w') as f:
        for record in train_records:
            f.write(record)
    
    with open('/home/Nema/UniCRS_GraphRAG/data/redial/valid_data_raw.jsonl', 'w') as f:
        for record in valid_records:
            f.write(record)

if __name__ == "__main__":
    process_files()