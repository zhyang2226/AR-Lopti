import pandas as pd
import argparse

def merge_parquet_files(file1, file2, output_file, merge_type="row"):

    print(f"Loading {file1}...")
    df1 = pd.read_parquet(file1)

    print(f"Loading {file2}...")
    df2 = pd.read_parquet(file2)

    merged_df = pd.concat([df1, df2], axis=0)

    print(f"Saving merged dataset to {output_file}...")
    import os
    output_path = os.path.dirname(output_file)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    merged_df.to_parquet(output_file, index=False)
    print("Merge complete!")
    print("Total rows in merged dataset:", len(merged_df))
    print("Total columns in merged dataset:", len(merged_df.columns))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two Parquet files into one.")
    parser.add_argument("--file1", type=str, help="Path to the first Parquet file.")
    parser.add_argument("--file2", type=str, help="Path to the second Parquet file.")
    parser.add_argument("--output_file", type=str, help="Path to the output Parquet file.")

    args = parser.parse_args()

    merge_parquet_files(args.file1, args.file2, args.output_file)

# python dataset_merge.py  --file1 ./data/kk/instruct/merge_34ppl/train.parquet --file2 ./data/kk/instruct/5ppl/train.parquet --output_file ./data/kk/instruct/merge_345ppl/train.parquet
# python dataset_merge.py  --file1 ./data/kk/instruct/merge_34ppl/test.parquet --file2 ./data/kk/instruct/5ppl/test.parquet --output_file ./data/kk/instruct/merge_345ppl/test.parquet

# python dataset_merge.py  --file1 ./data/kk/instruct/merge_345ppl/train.parquet --file2 ./data/kk/instruct/6ppl/train.parquet --output_file ./data/kk/instruct/merge_3456ppl/train.parquet
# python dataset_merge.py  --file1 ./data/kk/instruct/merge_345ppl/test.parquet --file2 ./data/kk/instruct/6ppl/test.parquet --output_file ./data/kk/instruct/merge_3456ppl/test.parquet

# python dataset_merge.py  --file1 ./data/kk/instruct/merge_3456ppl/train.parquet --file2 ./data/kk/instruct/7ppl/train.parquet --output_file ./data/kk/instruct/merge_34567ppl/train.parquet
# python dataset_merge.py  --file1 ./data/kk/instruct/merge_3456ppl/test.parquet --file2 ./data/kk/instruct/7ppl/test.parquet --output_file ./data/kk/instruct/merge_34567ppl/test.parquet
