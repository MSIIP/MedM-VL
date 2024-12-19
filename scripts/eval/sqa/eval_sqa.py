import argparse
import json
from tqdm import tqdm


def main(args):
    with open(args.answer_path, "r") as f:
        answer_list = json.load(f)
    with open(args.predict_path, "r") as f:
        predict_list = json.load(f)

    sqa_results = {}
    for answer_dict, predict in tqdm(zip(answer_list, predict_list), total=len(answer_list)):
        answer = answer_dict["answer"].strip()
        predict = predict.strip()
        sqa_results[answer_dict["id"]] = {
            "question": answer_dict["conversations"][0]["value"].strip(),
            "answer": answer,
            "predict": predict,
        }

    with open(args.dataset_split, "r") as f:
        dataset_split = json.load(f)
    dataset_split = dataset_split["test"]

    correct_list = []
    incorrect_list = []
    for question_id in tqdm(dataset_split):
        question = sqa_results[question_id]["question"]
        answer = sqa_results[question_id]["answer"]
        predict = sqa_results[question_id]["predict"]
        is_multimodal = "<image>" in question

        result = {
            "id": question_id,
            "is_multimodal": is_multimodal,
            "predict": predict,
        }

        if f"{answer}." in predict or f" {answer} " in (" " + predict + " "):
            correct_list.append(result)
        else:
            incorrect_list.append(result)

    correct = len(correct_list)
    total = len(correct_list) + len(incorrect_list)
    mm_correct = len([x for x in correct_list if x["is_multimodal"]])
    mm_incorrect = len([x for x in incorrect_list if x["is_multimodal"]])
    mm_total = mm_correct + mm_incorrect
    print(f'Total: {total}, Correct: {correct}, Accuracy: {correct / total * 100:.2f}%, IMG-Accuracy: {mm_correct / mm_total * 100:.2f}%')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer_path", type=str, default=None)
    parser.add_argument("--predict_path", type=str, default=None)
    parser.add_argument("--dataset_split", type=str, default=None)
    args = parser.parse_args()

    main(args)
