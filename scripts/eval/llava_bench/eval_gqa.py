import argparse
import json
from tqdm import tqdm


def main(args):
    with open(args.question_path, "r") as f:
        question_list = json.load(f)
    with open(args.predict_path, "r") as f:
        predict_list = json.load(f)

    results = []
    for question_dict, predict in tqdm(zip(question_list, predict_list), total=len(question_list)):
        questionId = question_dict["id"]
        prediction = predict.strip().lower()
        results.append(dict(questionId=questionId, prediction=prediction))

    with open(args.result_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_path", type=str, default=None)
    parser.add_argument("--predict_path", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    args = parser.parse_args()

    main(args)
