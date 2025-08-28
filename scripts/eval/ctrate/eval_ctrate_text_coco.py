import argparse
import json
from tqdm import tqdm

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor


def main(args):
    with open(args.answer_path, "r") as f:
        answer_list = json.load(f)
    with open(args.predict_path, "r") as f:
        predict_list = json.load(f)

    metric_list = ["bleu1", "bleu2", "bleu3", "bleu4", "rougeL", "meteor", "cider"]
    bleu = Bleu()
    rouge = Rouge()
    meteor = Meteor()
    cider = Cider()

    cases = []
    for answer_dict, predict in tqdm(zip(answer_list, predict_list), total=len(answer_list)):
        answer = answer_dict["answer"].strip()
        predict = predict.strip()
        if len(predict) == 0:
            cases.append({
                "id": answer_dict["id"],
                "answer": answer,
                "predict": predict,
                "bleu1": 0,
                "bleu2": 0,
                "bleu3": 0,
                "bleu4": 0,
                "rougeL": 0,
                "meteor": 0,
                "cider": 0,
            })
            continue

        bleu_score, _ = bleu.compute_score({answer_dict["id"]: [answer]}, {answer_dict["id"]: [predict]}, verbose=0)
        rouge_score, _ = rouge.compute_score({answer_dict["id"]: [answer]}, {answer_dict["id"]: [predict]})
        meteor_score, _ = meteor.compute_score({answer_dict["id"]: [answer]}, {answer_dict["id"]: [predict]})
        cider_score, _ = cider.compute_score({answer_dict["id"]: [answer]}, {answer_dict["id"]: [predict]})

        cases.append({
            "id": answer_dict["id"],
            "answer": answer,
            "predict": predict,
            "bleu1": bleu_score[0],
            "bleu2": bleu_score[1],
            "bleu3": bleu_score[2],
            "bleu4": bleu_score[3],
            "rougeL": rouge_score,
            "meteor": meteor_score,
            "cider": cider_score,
        })

    # calculate average scores
    results = {}
    for metric_name in metric_list:
        metric_scores = [r[metric_name] for r in cases]
        results[f"{metric_name}_avg"] = sum(metric_scores) / len(metric_scores)

    # calculate overall scores
    gts = {r["id"]: [r["answer"]] for r in cases}
    res = {r["id"]: [r["predict"]] for r in cases}
    bleu_score, _ = bleu.compute_score(gts, res, verbose=0)
    rouge_score, _ = rouge.compute_score(gts, res)
    meteor_score, _ = meteor.compute_score(gts, res)
    cider_score, _ = cider.compute_score(gts, res)
    results["bleu1"] = bleu_score[0]
    results["bleu2"] = bleu_score[1]
    results["bleu3"] = bleu_score[2]
    results["bleu4"] = bleu_score[3]
    results["rougeL"] = rouge_score
    results["meteor"] = meteor_score
    results["cider"] = cider_score

    for metric_name in metric_list:
        print(f"{metric_name}_avg:", results[f"{metric_name}_avg"])
        print(f"{metric_name}:", results[metric_name])

    results["cases"] = cases
    with open(args.result_path, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer_path", type=str, default=None)
    parser.add_argument("--predict_path", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    args = parser.parse_args()

    main(args)
