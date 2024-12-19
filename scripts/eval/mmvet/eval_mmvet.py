import argparse
import json
from tqdm import tqdm


def main(args):
    with open(args.predict_path, "r") as f:
        predict_list = json.load(f)

    results = {}
    for idx, predict in tqdm(enumerate(predict_list)):
        results[f"v1_{idx}"] = predict.strip()

    with open(args.result_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_path", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    args = parser.parse_args()

    main(args)
