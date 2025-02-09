
# This script is used to get a sample dataset from a large dataset.

import json

# input_file = '/Volumes/T7_Shield/Datasets/relationships.json'
input_file = '/Volumes/T7_Shield/Datasets/image_data.json'
# output_file = 'data/relationships.json'
output_file = 'data/image_data.json'
sample_size = 1000

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

if isinstance(data, list):
    sample_data = data[:sample_size]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=4)

    print(f"Get {sample_size} sample datas to {output_file}")
else:
    print("Error: The input data is not a list.")