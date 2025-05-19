import pandas as pd
import argparse
import os

def extract_prompts(item):
    if 'problem' in item:
        return [{"content": item['problem'], "role": "user"}]
    elif 'question' in item:
        return [{"content": item['question'], "role": "user"}]
    else:
        raise ValueError("Item does not contain 'problem' or 'question' key.")
    
def extract_answers(item):
    if 'solution' in item:
        return {"ground_truth": item['solution']}
    elif 'answer' in item:
        return {"ground_truth": item['answer']}
    else:
        raise ValueError("Item does not contain 'solution' or 'answer' key.")

def from_jsonl_to_parquet(json_file, partuet_file):
    if "amc" in json_file:
        data_source = "amc"
    elif "aime" in json_file:
        data_source = "aime"
    else:
        raise ValueError("Unsupported data source in filename.")
    new_data = []
    # Read the JSONL file into a list of dictionaries
    with open(json_file, 'r', encoding='utf-8') as file:
        for item in file:
            new_item = {"data_source": data_source,
                        "prompt": extract_prompts(eval(item.strip())),
                        "reward_model": extract_answers(eval(item.strip()))}
            new_data.append(new_item)
    parquet_df = pd.DataFrame(new_data)
    parquet_df.to_parquet(partuet_file, index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change the format from jsonl to parquet.")
    parser.add_argument("--json_file", type=str, help="Path to the first Parquet file.")
    parser.add_argument("--partuet_file", type=str, help="Path to the output Parquet file.")

    args = parser.parse_args()

    from_jsonl_to_parquet(args.json_file, args.partuet_file)

# python dataset_format.py --json_file ./data/aime/aime_2021_2024.jsonl --partuet_file ./data/aime/aime_2021_2024.parquet
# python dataset_format.py --json_file ./data/amc/amc.jsonl --partuet_file ./data/amc/amc.parquet