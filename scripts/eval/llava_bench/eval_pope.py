import argparse
import json
import os
import os.path as osp
from tqdm import tqdm


def eval_pope(answers, label_file):
    label_list = [json.loads(q)["label"] for q in open(label_file, "r")]

    for answer in answers:
        text = answer["text"]

        # Only keep the first sentence
        if text.find(".") != -1:
            text = text.split(".")[0]

        text = text.replace(",", "")
        words = text.split(" ")
        if "No" in words or "not" in words or "no" in words:
            answer["text"] = "no"
        else:
            answer["text"] = "yes"

    for i in range(len(label_list)):
        if label_list[i] == "no":
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer["text"] == "no":
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print("TP\tFP\tTN\tFN\t")
    print("{}\t{}\t{}\t{}".format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print("Accuracy: {}".format(acc))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1 score: {}".format(f1))
    print("Yes ratio: {}".format(yes_ratio))
    print("%.3f, %.3f, %.3f, %.3f, %.3f" % (f1, acc, precision, recall, yes_ratio))


def main(args):
    with open(args.question_path, "r") as f:
        question_list = json.load(f)
    with open(args.predict_path, "r") as f:
        predict_list = json.load(f)

    results = []
    for question_dict, predict in tqdm(zip(question_list, predict_list), total=len(question_list)):
        results.append(
            {
                "id": question_dict["id"],
                "text": predict.strip(),
                "category": question_dict["category"],
            }
        )
    questions = {q["id"]: q for q in question_list}

    for file in os.listdir(args.answer_dir):
        category = file[10:-5]
        cur_answers = [x for x in results if questions[x["id"]]["category"] == category]
        print("Category: {}, # samples: {}".format(category, len(cur_answers)))
        eval_pope(cur_answers, osp.join(args.answer_dir, file))
        print("====================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_path", type=str, default=None)
    parser.add_argument("--predict_path", type=str, default=None)
    parser.add_argument("--answer_dir", type=str, default=None)
    args = parser.parse_args()

    main(args)
