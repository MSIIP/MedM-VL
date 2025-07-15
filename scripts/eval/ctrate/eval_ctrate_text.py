import argparse
import json
from tqdm import tqdm

import evaluate


def main(args):
    with open(args.answer_path, "r") as f:
        answer_list = json.load(f)
    with open(args.predict_path, "r") as f:
        predict_list = json.load(f)

    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    bertscore = evaluate.load("bertscore")

    results = []
    for answer_dict, predict in tqdm(zip(answer_list, predict_list), total=len(answer_list)):
        answer = answer_dict["answer"].strip()
        predict = predict.strip()
        if len(predict) == 0:
            results.append({
                "id": answer_dict["id"],
                "answer": answer,
                "predict": predict,
                "bleu1": 0,
                "bleu2": 0,
                "bleu3": 0,
                "bleu4": 0,
                "rouge1": 0,
                "rougeL": 0,
                "meteor": 0,
                "bert_f1": 0,
            })
            continue

        bleu1_score = bleu.compute(predictions=[predict], references=[answer], max_order=1)
        bleu2_score = bleu.compute(predictions=[predict], references=[answer], max_order=2)
        bleu3_score = bleu.compute(predictions=[predict], references=[answer], max_order=3)
        bleu4_score = bleu.compute(predictions=[predict], references=[answer], max_order=4)
        rouge_score = rouge.compute(predictions=[predict], references=[answer])
        meteor_score = meteor.compute(predictions=[predict], references=[answer])
        bert_score = bertscore.compute(predictions=[predict], references=[answer], lang="en")

        results.append({
            "id": answer_dict["id"],
            "answer": answer,
            "predict": predict,
            "bleu1": bleu1_score["bleu"],
            "bleu2": bleu2_score["bleu"],
            "bleu3": bleu3_score["bleu"],
            "bleu4": bleu4_score["bleu"],
            "rouge1": rouge_score["rouge1"],
            "rougeL": rouge_score["rougeL"],
            "meteor": meteor_score["meteor"],
            "bert_f1": sum(bert_score["f1"]) / len(bert_score["f1"]),
        })

    with open(args.result_path, "w") as f:
        json.dump(results, f)

    # calculate average scores
    results_avg = {}
    metric_list = ["bleu1", "bleu2", "bleu3", "bleu4", "rouge1", "rougeL", "meteor", "bert_f1"]
    for metric_name in metric_list:
        metric_scores = [r[metric_name] for r in results]
        print(f"{metric_name}: {sum(metric_scores) / len(metric_scores)}")
        results_avg[metric_name] = sum(metric_scores) / len(metric_scores)

    with open(args.result_path.replace(".json", "_avg.json"), "w") as f:
        json.dump(results_avg, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer_path", type=str, default=None)
    parser.add_argument("--predict_path", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    args = parser.parse_args()

    main(args)
