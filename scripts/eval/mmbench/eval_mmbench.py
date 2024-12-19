import argparse
import json
from tqdm import tqdm
import pandas as pd


def main(args):
    with open(args.predict_path, "r") as f:
        predict_list = json.load(f)

    df = pd.read_table(args.answer_path)
    cur_df = df.copy()
    cur_df = cur_df.drop(columns=["hint", "category", "source", "image", "comment", "l2-category"])
    cur_df.insert(6, "prediction", None)

    count = 0
    for i, predict in tqdm(enumerate(predict_list)):
        cur_df.loc[i, "prediction"] = predict.strip()
        # print(cur_df.loc[i, 'answer'])
        # print(pred['outputs'][1:2])

        if predict.strip() == cur_df.loc[i, "answer"].strip():
            count += 1

    cur_df.to_excel(args.result_path, index=False, engine="openpyxl")
    print(len(predict_list))
    print(count / len(predict_list))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_path", type=str, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--answer_path", type=str, default=None)
    args = parser.parse_args()

    main(args)
