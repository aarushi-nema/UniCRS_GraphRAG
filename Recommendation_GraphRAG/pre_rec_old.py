import pandas as pd
import json
import time
import subprocess


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
        else:
            print("Command failed with return code:", process.returncode)
        
        if process.stdout:
            print("Command output:", process.stdout)
        if process.stderr:
            print("Command errors:", process.stderr)

        ## READ CURRENT GENERATED FILES: /home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG/output/create_final_entities.parquet and /home/Nema/UniCRS_GraphRAG/Recommendation_GraphRAG/output/create_final_relationships.parquet

        # Read files
        df = pd.read_parquet()

        
        time.sleep(2)