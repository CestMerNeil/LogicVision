
# This script is used to get a sample dataset from a large dataset.

import json

input_file = '/root/autodl-tmp/relationships_full.json'
# input_file = '/root/autodl-tmp/image_data_full.json'
output_file = '/root/autodl-tmp/relationships.json'
# output_file = '/root/autodl-tmp/image_data.json'
sample_size = 10000

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

if isinstance(data, list):
    sample_data = data[:sample_size]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=4)

    print(f"Get {sample_size} sample datas to {output_file}")
else:
    print("Error: The input data is not a list.")