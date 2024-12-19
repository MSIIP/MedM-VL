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

    m3d_cap_results = []
    for answer_dict, predict in tqdm(zip(answer_list, predict_list), total=len(answer_list)):
        answer = answer_dict["answer"].strip()
        predict = predict.strip()
        if len(predict) == 0:
            m3d_cap_results.append({
                "id": answer_dict["id"],
                "answer": answer,
                "predict": predict,
                "bleu": 0,
                "rouge1": 0,
                "meteor": 0,
                "bert_f1": 0,
            })
            continue

        bleu_score = bleu.compute(predictions=[predict], references=[answer], max_order=1)
        rouge_score = rouge.compute(predictions=[predict], references=[answer], rouge_types=["rouge1"])
        meteor_score = meteor.compute(predictions=[predict], references=[answer])
        bert_score = bertscore.compute(predictions=[predict], references=[answer], lang="en")

        m3d_cap_results.append({
            "id": answer_dict["id"],
            "answer": answer,
            "predict": predict,
            "bleu": bleu_score["bleu"],
            "rouge1": rouge_score["rouge1"],
            "meteor": meteor_score["meteor"],
            "bert_f1": sum(bert_score["f1"]) / len(bert_score["f1"]),
        })

    # calculate average scores
    bleu_scores = [x["bleu"] for x in m3d_cap_results]
    rouge1_scores = [x["rouge1"] for x in m3d_cap_results]
    meteor_scores = [x["meteor"] for x in m3d_cap_results]
    bert_f1_scores = [x["bert_f1"] for x in m3d_cap_results]
    print(f"BLEU: {sum(bleu_scores) / len(bleu_scores)}")
    print(f"ROUGE-1: {sum(rouge1_scores) / len(rouge1_scores)}")
    print(f"METEOR: {sum(meteor_scores) / len(meteor_scores)}")
    print(f"BertScore: {sum(bert_f1_scores) / len(bert_f1_scores)}")

    with open(args.result_path, "w") as f:
        json.dump(m3d_cap_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer_path", type=str, default=None)
    parser.add_argument("--predict_path", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    args = parser.parse_args()

    main(args)
