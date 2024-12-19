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

    m3d_vqa_open_results = []
    for answer_dict, predict in tqdm(zip(answer_list, predict_list), total=len(answer_list)):
        answer = answer_dict["answer"].strip()
        predict = predict.strip()
        if len(predict) == 0:
            m3d_vqa_open_results.append({
                "id": answer_dict["id"],
                "answer": answer,
                "predict": predict,
                "bleu": 0,
                "rouge1": 0,
                "meteor": 0,
                "bert_f1": 0,
                "question_type": answer_dict["question_type"],
            })
            continue

        bleu_score = bleu.compute(predictions=[predict], references=[answer], max_order=1)
        rouge_score = rouge.compute(predictions=[predict], references=[answer], rouge_types=["rouge1"])
        meteor_score = meteor.compute(predictions=[predict], references=[answer])
        bert_score = bertscore.compute(predictions=[predict], references=[answer], lang="en")

        m3d_vqa_open_results.append({
            "id": answer_dict["id"],
            "answer": answer,
            "predict": predict,
            "bleu": bleu_score["bleu"],
            "rouge1": rouge_score["rouge1"],
            "meteor": meteor_score["meteor"],
            "bert_f1": sum(bert_score["f1"]) / len(bert_score["f1"]),
            "question_type": answer_dict["question_type"],
        })

    metrics = ["bleu", "rouge1", "meteor", "bert_f1"]
    for metric in metrics:
        metric_list = []
        qtypes = sorted(set([r["question_type"] for r in m3d_vqa_open_results]))
        for qtype in qtypes:
            scores = [r[metric] for r in m3d_vqa_open_results if r["question_type"] == qtype]
            avg_score = sum(scores) / len(scores)
            metric_list.append(avg_score)
            print(f"Question type: {qtype}, Average {metric}: {avg_score}")
        print(f"Average {metric}: {sum(metric_list) / len(metric_list)}")

        scores = [r[metric] for r in m3d_vqa_open_results]
        avg_score = sum(scores) / len(scores)
        print(f"{metric}: {avg_score}\n")

    with open(args.result_path, "w") as f:
        json.dump(m3d_vqa_open_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer_path", type=str, default=None)
    parser.add_argument("--predict_path", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    args = parser.parse_args()

    main(args)
