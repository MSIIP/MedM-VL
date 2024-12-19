import argparse
import json
from tqdm import tqdm


def main(args):
    with open(args.answer_path, "r") as f:
        answer_list = json.load(f)
    with open(args.predict_path, "r") as f:
        predict_list = json.load(f)

    m3d_vqa_close_results = []
    for answer_dict, predict in tqdm(zip(answer_list, predict_list), total=len(answer_list)):
        answer = answer_dict["answer"].strip()
        predict = predict.strip()

        correct = (answer_dict["answer_choice"] + ".") in predict
        m3d_vqa_close_results.append({
            "id": answer_dict["id"],
            "answer": answer,
            "predict": predict,
            "correct": correct,
            "question_type": answer_dict["question_type"],
        })

    metrics = ["correct"]
    for metric in metrics:
        metric_list = []
        qtypes = sorted(set([r["question_type"] for r in m3d_vqa_close_results]))
        for qtype in qtypes:
            scores = [r[metric] for r in m3d_vqa_close_results if r["question_type"] == qtype]
            avg_score = sum(scores) / len(scores)
            metric_list.append(avg_score)
            print(f"Question type: {qtype}, Average {metric}: {avg_score}")
        print(f"Average {metric}: {sum(metric_list) / len(metric_list)}")

        scores = [r[metric] for r in m3d_vqa_close_results]
        avg_score = sum(scores) / len(scores)
        print(f"{metric}: {avg_score}\n")

    with open(args.result_path, "w") as f:
        json.dump(m3d_vqa_close_results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer_path", type=str, default=None)
    parser.add_argument("--predict_path", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    args = parser.parse_args()

    main(args)
