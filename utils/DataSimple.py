import json


def main():
    """Get a sample dataset from a large dataset.

    Loads data from an input JSON file, extracts a sample of a specified size
    if the data is a list, and writes the sample to an output JSON file.

    Raises:
        ValueError: If the input data is not a list.
    """
    input_file = "/root/autodl-tmp/relationships_full.json"
    output_file = "/root/autodl-tmp/relationships.json"
    sample_size = 1000

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        sample_data = data[:sample_size]
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=4)
        print(f"Get {sample_size} sample datas to {output_file}")
    else:
        raise ValueError("The input data is not a list.")


if __name__ == "__main__":
    main()
