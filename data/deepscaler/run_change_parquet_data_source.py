import pandas as pd
import pyarrow as pa
import os
import argparse

import pyarrow.parquet as pq

def change_data_source(input_parquet_path, output_parquet_path, new_data_source):
    """
    Read a Parquet file, change the data_source column to a new value, and write to a new Parquet file.
    
    Args:
        input_parquet_path (str): Path to the input Parquet file
        output_parquet_path (str): Path to save the modified Parquet file
        new_data_source (str): New value for the data_source column
    """
    try:
        # Read the Parquet file into a pandas DataFrame
        df = pd.read_parquet(input_parquet_path)
        
        # Check if 'data_source' column exists
        if 'data_source' in df.columns:
            # Modify the data_source column
            df['data_source'] = new_data_source
            
            # Write the modified DataFrame to a new Parquet file
            df.to_parquet(output_parquet_path, index=False)
            
            print(f"Successfully modified data_source to '{new_data_source}'")
            print(f"New Parquet file saved to: {output_parquet_path}")
        else:
            print("Warning: 'data_source' column not found in the Parquet file.")
            
    except Exception as e:
        print(f"Error processing Parquet file: {e}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Change data_source value in a Parquet file")
    parser.add_argument("--input", required=True, help="Input Parquet file path")
    parser.add_argument("--output", help="Output Parquet file path")
    parser.add_argument("--data-source", required=True, help="New value for data_source")
    
    args = parser.parse_args()
    
    # If output file path is not specified, use input filename with a prefix
    if not args.output:
        input_dir = os.path.dirname(args.input)
        input_filename = os.path.basename(args.input)
        args.output = os.path.join(input_dir, f"modified_{input_filename}")
    
    change_data_source(args.input, args.output, args.data_source)