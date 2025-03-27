import argparse
import json
import os
import os.path as osp
from collections import defaultdict
from tqdm import tqdm


def get_gt(data_path):
    GT = {}
    for category in os.listdir(data_path):
        category_dir = osp.join(data_path, category)
        if not osp.isdir(category_dir):
            continue
        if osp.exists(osp.join(category_dir, "images")):
            image_path = osp.join(category_dir, "images")
            qa_path = osp.join(category_dir, "questions_answers_YN")
        else:
            image_path = qa_path = category_dir
        assert osp.isdir(image_path), image_path
        assert osp.isdir(qa_path), qa_path
        for file in os.listdir(qa_path):
            if not file.endswith(".txt"):
                continue
            for line in open(osp.join(qa_path, file)):
                question, answer = line.strip().split("\t")
                GT[(category, file, question)] = answer
    return GT


def main(args):
    with open(args.question_path, "r") as f:
        question_list = json.load(f)
    with open(args.predict_path, "r") as f:
        predict_list = json.load(f)

    results = defaultdict(list)
    for question_dict, predict in tqdm(zip(question_list, predict_list), total=len(question_list)):
        category = question_dict["id"].split("/")[0]
        file = question_dict["id"].split("/")[-1].split(".")[0] + ".txt"
        question = question_dict["conversations"][0]["value"][8:]
        results[category].append((file, question, predict.strip()))

    GT = get_gt(args.answer_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    for category, cate_tups in results.items():
        with open(osp.join(args.output_dir, f"{category}.txt"), "w") as fp:
            for file, prompt, answer in cate_tups:
                if "Answer the question using a single word or phrase." in prompt:
                    prompt = prompt.replace("Answer the question using a single word or phrase.", "").strip()
                if "Please answer yes or no." not in prompt:
                    prompt = prompt + " Please answer yes or no."
                    if (category, file, prompt) not in GT:
                        prompt = prompt.replace(" Please answer yes or no.", "  Please answer yes or no.")
                gt_ans = GT[category, file, prompt]
                tup = file, prompt, gt_ans, answer
                fp.write("\t".join(tup) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_path", type=str, default=None)
    parser.add_argument("--predict_path", type=str, default=None)
    parser.add_argument("--answer_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    main(args)
