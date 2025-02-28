import json

def transform_messages(conversation):
    """Replace movie IDs with actual movie names."""
    movie_mentions = conversation.get("movieMentions", {})
    transformed_messages = []
    
    for message in conversation["messages"]:
        text = message["text"]
        # Replace all movie IDs with their names, skipping None values
        for movie_id, movie_name in movie_mentions.items():
            if movie_name is not None:  # Only replace if we have a valid movie name
                text = text.replace(f"@{movie_id}", movie_name)
            else:
                # Keep the original ID if no name is available
                text = text.replace(f"@{movie_id}", f"movie_{movie_id}")
        
        transformed_message = {
            "text": text,
            "senderWorkerId": message["senderWorkerId"]
        }
        transformed_messages.append(transformed_message)
    
    return transformed_messages, conversation["initiatorWorkerId"]

def format_dialogue(messages, initiator_id):
    """Format messages into dialogue string."""
    dialogue = []
    for msg in messages:
        # Identify if sender is initiator or respondent
        sender_type = "User" if msg["senderWorkerId"] == initiator_id else "Recommender"
        dialogue.append(f"{sender_type}: {msg['text']}")
    
    # Return dialogue without curly braces
    return " | ".join(dialogue)

def process_file(input_path, output_path):
    """Process the entire file from original format to final dialogue format."""
    try:
        with open(input_path, 'r', encoding='utf-8') as f, open(output_path, 'w', encoding='utf-8') as out_f:
            line_count = 0
            error_count = 0
            for line in f:
                if line.strip():
                    try:
                        # Load original conversation
                        conversation = json.loads(line)
                        
                        # Transform movie IDs to names
                        transformed_messages, initiator_id = transform_messages(conversation)
                        
                        # Format into final dialogue
                        final_dialogue = format_dialogue(transformed_messages, initiator_id)
                        
                        # Write to output file
                        out_f.write(final_dialogue + '\n')
                        line_count += 1
                        
                        # Print progress every 1000 lines
                        if line_count % 1000 == 0:
                            print(f"Processed {line_count} conversations...")
                            
                    except Exception as e:
                        error_count += 1
                        print(f"Error processing conversation: {str(e)}")
                        continue
            
            print(f"Processing complete. Successfully processed {line_count} conversations.")
            if error_count > 0:
                print(f"Encountered {error_count} errors during processing.")
                
    except Exception as e:
        print(f"Error opening files: {str(e)}")
        raise

def main():
    file_pairs = [
        {
            'input': '/home/Nema/UniCRS_GraphRAG/data/redial/test_data_raw.jsonl',
            'output': '/home/Nema/UniCRS_GraphRAG/data/redial/test_data.txt'
        },
        {
            'input': '/home/Nema/UniCRS_GraphRAG/data/redial/train_data_raw.jsonl',
            'output': '/home/Nema/UniCRS_GraphRAG/data/redial/train_data.txt'
        },
        {
            'input': '/home/Nema/UniCRS_GraphRAG/data/redial/valid_data_raw.jsonl',
            'output': '/home/Nema/UniCRS_GraphRAG/data/redial/valid_data.txt'
        }
    ]
    
    # First process all files
    for file_pair in file_pairs:
        print(f"\nProcessing {file_pair['input']}...")
        process_file(file_pair['input'], file_pair['output'])
    
    # Then combine all txt files into one
    combined_output = '/home/Nema/UniCRS_GraphRAG/data/redial/graphrag_data.txt'
    with open(combined_output, 'w', encoding='utf-8') as outfile:
        for file_pair in file_pairs:
            try:
                with open(file_pair['output'], 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
            except Exception as e:
                print(f"Error reading {file_pair['output']}: {str(e)}")
    
    print(f"\nAll files combined into {combined_output}")

if __name__ == "__main__":
    main()