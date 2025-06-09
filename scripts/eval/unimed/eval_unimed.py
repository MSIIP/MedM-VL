import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge as Rouge_coco
from rouge import Rouge


model_path = "work_dirs/MedM-VL-2D-3B-en"
gt_ori = '/ssd/zhux/uni_med/uni_med/datasets/datasets/eval'
path = os.path.join(model_path, 'eval')
metric = os.path.join(model_path, 'eval_results')
os.makedirs(metric, exist_ok=True)
results = {}

articles = ["a", "an", "the"]
contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}
manual_map = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")


def computeIoU(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    intersection_x1 = max(x1, x3)
    intersection_y1 = max(y1, y3)
    intersection_x2 = min(x2, x4)
    intersection_y2 = min(y2, y4)
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)
    bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    bbox2_area = (x4 - x3 + 1) * (y4 - y3 + 1)
    union_area = bbox1_area + bbox2_area - intersection_area
    iou = intersection_area / union_area
    return iou


def normalize_word(token):
    _token = token
    for p in punct:
        if (p + " " in token or " " + p in token) or (
            re.search(comma_strip, token) != None
        ):
            _token = _token.replace(p, "")
        else:
            _token = _token.replace(p, " ")
    token = period_strip.sub("", _token, re.UNICODE)

    _token = []
    temp = token.lower().split()
    for word in temp:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            _token.append(word)
    for i, word in enumerate(_token):
        if word in contractions:
            _token[i] = contractions[word]
    token = " ".join(_token)
    token = token.replace(",", "")
    return token


def split_sentence(sentence, n):
    words = defaultdict(int)
    tmp_sentence = sentence
    tmp_sentence = tmp_sentence.lower()
    tmp_sentence = tmp_sentence.strip().split()
    length = len(tmp_sentence)
    for i in range(length - n + 1):
        tmp_words = " ".join(tmp_sentence[i: i + n])
        if tmp_words:
            words[tmp_words] += 1
    return words


def json_trans(data, normalize):
    gt = {}
    pred ={}
    for i in range(len(data)):
        answer_pred = data[i]["answer_pred"]
        answer_gt = data[i]["answer_gt"]

        if normalize:
            answer_pred = normalize_word(answer_pred)
            answer_gt = normalize_word(answer_gt)
        
        pred[i] = answer_pred
        gt[i] = answer_gt

    return pred, gt


def to_coco_format(caption):
    coco = {}
    for k, v in caption.items():
        coco[k] = [{'caption': v}]
    return coco


def pycocoeval(data, scorers, normalize):
    tokenizer = PTBTokenizer()
    pred, gt=json_trans(data, normalize)
    pred = to_coco_format(pred)
    gt = to_coco_format(gt)
    pred = tokenizer.tokenize(pred)
    gt = tokenizer.tokenize(gt)

    eval = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(gt, pred)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval[m] = sc
        else:
            eval[method] = score
    
    return eval


def calculate_bleu_nltk(data):
    bleu1_nltk_no_norm_list = []
    bleu2_nltk_no_norm_list = []
    bleu3_nltk_no_norm_list = []
    bleu4_nltk_no_norm_list = []

    bleu1_nltk_sm_no_norm_list = []
    bleu2_nltk_sm_no_norm_list = []
    bleu3_nltk_sm_no_norm_list = []
    bleu4_nltk_sm_no_norm_list = []

    bleu1_nltk_norm_list = []
    bleu2_nltk_norm_list = []
    bleu3_nltk_norm_list = []
    bleu4_nltk_norm_list = []

    bleu1_nltk_sm_norm_list = []
    bleu2_nltk_sm_norm_list = []
    bleu3_nltk_sm_norm_list = []
    bleu4_nltk_sm_norm_list = []

    smoothing_function = SmoothingFunction()

    for ann in data:
        answer_gt = ann["answer_gt"].split()
        answer_pred = ann["answer_pred"].split()
    
        bleu1_nltk_no_norm_list.append(sentence_bleu([answer_gt], answer_pred, weights=(1, 0, 0, 0)))
        bleu2_nltk_no_norm_list.append(sentence_bleu([answer_gt], answer_pred, weights=(1./2., 1./2.)))
        bleu3_nltk_no_norm_list.append(sentence_bleu([answer_gt], answer_pred, weights=(1./3., 1./3., 1./3.)))
        bleu4_nltk_no_norm_list.append(sentence_bleu([answer_gt], answer_pred, weights=(1./4., 1./4., 1./4., 1./4.)))

        bleu1_nltk_sm_no_norm_list.append(sentence_bleu([answer_gt], answer_pred, weights=(1, 0, 0, 0), smoothing_function=smoothing_function.method1))
        bleu2_nltk_sm_no_norm_list.append(sentence_bleu([answer_gt], answer_pred, weights=(1./2., 1./2.), smoothing_function=smoothing_function.method1))
        bleu3_nltk_sm_no_norm_list.append(sentence_bleu([answer_gt], answer_pred, weights=(1./3., 1./3., 1./3.), smoothing_function=smoothing_function.method1))
        bleu4_nltk_sm_no_norm_list.append(sentence_bleu([answer_gt], answer_pred, weights=(1./4., 1./4., 1./4., 1./4.), smoothing_function=smoothing_function.method1))

        answer_gt = normalize_word(ann["answer_gt"]).split()
        answer_pred = normalize_word(ann["answer_pred"]).split()

        bleu1_nltk_norm_list.append(sentence_bleu([answer_gt], answer_pred, weights=(1, 0, 0, 0)))
        bleu2_nltk_norm_list.append(sentence_bleu([answer_gt], answer_pred, weights=(1./2., 1./2.)))
        bleu3_nltk_norm_list.append(sentence_bleu([answer_gt], answer_pred, weights=(1./3., 1./3., 1./3.)))
        bleu4_nltk_norm_list.append(sentence_bleu([answer_gt], answer_pred, weights=(1./4., 1./4., 1./4., 1./4.)))

        bleu1_nltk_sm_norm_list.append(sentence_bleu([answer_gt], answer_pred, weights=(1, 0, 0, 0), smoothing_function=smoothing_function.method1))
        bleu2_nltk_sm_norm_list.append(sentence_bleu([answer_gt], answer_pred, weights=(1./2., 1./2.), smoothing_function=smoothing_function.method1))
        bleu3_nltk_sm_norm_list.append(sentence_bleu([answer_gt], answer_pred, weights=(1./3., 1./3., 1./3.), smoothing_function=smoothing_function.method1))
        bleu4_nltk_sm_norm_list.append(sentence_bleu([answer_gt], answer_pred, weights=(1./4., 1./4., 1./4., 1./4.), smoothing_function=smoothing_function.method1))

    bleu1_nltk_no_norm_list = np.array(bleu1_nltk_no_norm_list).mean()
    bleu2_nltk_no_norm_list = np.array(bleu2_nltk_no_norm_list).mean()
    bleu3_nltk_no_norm_list = np.array(bleu3_nltk_no_norm_list).mean()
    bleu4_nltk_no_norm_list = np.array(bleu4_nltk_no_norm_list).mean()

    bleu1_nltk_sm_no_norm_list = np.array(bleu1_nltk_sm_no_norm_list).mean()
    bleu2_nltk_sm_no_norm_list = np.array(bleu2_nltk_sm_no_norm_list).mean()
    bleu3_nltk_sm_no_norm_list = np.array(bleu3_nltk_sm_no_norm_list).mean()
    bleu4_nltk_sm_no_norm_list = np.array(bleu4_nltk_sm_no_norm_list).mean()

    bleu1_nltk_norm_list = np.array(bleu1_nltk_norm_list).mean()
    bleu2_nltk_norm_list = np.array(bleu2_nltk_norm_list).mean()
    bleu3_nltk_norm_list = np.array(bleu3_nltk_norm_list).mean()
    bleu4_nltk_norm_list = np.array(bleu4_nltk_norm_list).mean()

    bleu1_nltk_sm_norm_list = np.array(bleu1_nltk_sm_norm_list).mean()
    bleu2_nltk_sm_norm_list = np.array(bleu2_nltk_sm_norm_list).mean()
    bleu3_nltk_sm_norm_list = np.array(bleu3_nltk_sm_norm_list).mean()
    bleu4_nltk_sm_norm_list = np.array(bleu4_nltk_sm_norm_list).mean()

    return {
        'bleu1': {
            'nltk_no_norm': bleu1_nltk_no_norm_list,
            'nltk_sm_no_norm': bleu1_nltk_sm_no_norm_list,
            'nltk_norm': bleu1_nltk_norm_list,
            'nltk_sm_norm': bleu1_nltk_sm_norm_list
        },
        'bleu2': {
            'nltk_no_norm': bleu2_nltk_no_norm_list,
            'nltk_sm_no_norm': bleu2_nltk_sm_no_norm_list,
            'nltk_norm': bleu2_nltk_norm_list,
            'nltk_sm_norm': bleu2_nltk_sm_norm_list
        },
        'bleu3': {
            'nltk_no_norm': bleu3_nltk_no_norm_list,
            'nltk_sm_no_norm': bleu3_nltk_sm_no_norm_list,
            'nltk_norm': bleu3_nltk_norm_list,
            'nltk_sm_norm': bleu3_nltk_sm_norm_list
        },
        'bleu4': {
            'nltk_no_norm': bleu4_nltk_no_norm_list,
            'nltk_sm_no_norm': bleu4_nltk_sm_no_norm_list,
            'nltk_norm': bleu4_nltk_norm_list,
            'nltk_sm_norm': bleu4_nltk_sm_norm_list
        }
    }


def calculate_bleu_coco(data):
    scorers = [
        (Bleu(4), ["bleu1", "bleu2", "bleu3", "bleu4"]),
    ]

    no_norm_result = pycocoeval(data, scorers, normalize=False)
    norm_result = pycocoeval(data, scorers, normalize=True)

    return {
        'bleu1': {
            'coco_no_norm': no_norm_result['bleu1'],
            'coco_norm': norm_result['bleu1'],
        },
        'bleu2': {
            'coco_no_norm': no_norm_result['bleu2'],
            'coco_norm': norm_result['bleu2'],
        },
        'bleu3': {
            'coco_no_norm': no_norm_result['bleu3'],
            'coco_norm': norm_result['bleu3'],
        },
        'bleu4': {
            'coco_no_norm': no_norm_result['bleu4'],
            'coco_norm': norm_result['bleu4'],
        },
    }


def calculate_bleu(data):
    nltk_result = calculate_bleu_nltk(data)
    coco_result = calculate_bleu_coco(data)

    new_result = {}
    for bleu in set(nltk_result.keys()) | set(coco_result.keys()):
        new_result[bleu] = {}
        if bleu in nltk_result:
            new_result[bleu].update(nltk_result[bleu])
        if bleu in coco_result:
            new_result[bleu].update(coco_result[bleu])

    return new_result


def calculate_meteor(data):
    scorers = [
        (Meteor(),"meteor")
    ]

    no_norm_result = pycocoeval(data, scorers, normalize=False)
    norm_result = pycocoeval(data, scorers, normalize=True)

    return {
        'no_norm': no_norm_result['meteor'],
        'norm': norm_result['meteor'],
    }


def calculate_rouge_coco(data):
    scorers = [
        (Rouge_coco(), "rouge_l"),
    ]

    no_norm_result = pycocoeval(data, scorers, normalize=False)
    norm_result = pycocoeval(data, scorers, normalize=True)

    return {
        'rouge_l': {
            'coco_no_norm': no_norm_result['rouge_l'],
            'coco_norm': norm_result['rouge_l'],
        },
    }


def calculate_rouge_original(data):
    rouge = Rouge()

    rouge1_no_norm_list = []
    rouge2_no_norm_list = []
    rougel_no_norm_list = []

    rouge1_norm_list = []
    rouge2_norm_list = []
    rougel_norm_list = []

    for ann in data:
        answer_gt = ann["answer_gt"]
        answer_pred = ann["answer_pred"]
        
        if not answer_gt or not answer_pred:
            continue

        rouge_score_no_norm = rouge.get_scores(answer_pred, answer_gt)

        rouge1_no_norm_list.append(rouge_score_no_norm[0]["rouge-1"]["f"])
        rouge2_no_norm_list.append(rouge_score_no_norm[0]["rouge-2"]["f"])
        rougel_no_norm_list.append(rouge_score_no_norm[0]["rouge-l"]["f"])
        
        answer_pred = normalize_word(answer_pred)
        answer_gt = normalize_word(answer_gt)
        
        if not answer_gt or not answer_pred:
            continue

        rouge_score_norm = rouge.get_scores(answer_pred, answer_gt)

        rouge1_norm_list.append(rouge_score_norm[0]["rouge-1"]["f"])
        rouge2_norm_list.append(rouge_score_norm[0]["rouge-2"]["f"])
        rougel_norm_list.append(rouge_score_norm[0]["rouge-l"]["f"])
    
    rouge1_no_norm_list = np.array(rouge1_no_norm_list).mean()
    rouge2_no_norm_list = np.array(rouge2_no_norm_list).mean()
    rougel_no_norm_list = np.array(rougel_no_norm_list).mean()

    rouge1_norm_list = np.array(rouge1_norm_list).mean()
    rouge2_norm_list = np.array(rouge2_norm_list).mean()
    rougel_norm_list = np.array(rougel_norm_list).mean()

    return {
        'rouge_1': {
            'original_no_norm': rouge1_no_norm_list,
            'original_norm': rouge1_norm_list,
        },
        'rouge_2': {
            'original_no_norm': rouge2_no_norm_list,
            'original_norm': rouge2_norm_list,
        },
        'rouge_l': {
            'original_no_norm': rougel_no_norm_list,
            'original_norm': rougel_norm_list,
        },
    }


def calculate_rouge(data):
    original_result = calculate_rouge_original(data)
    coco_result = calculate_rouge_coco(data)

    new_result = {}
    for rouge in set(original_result.keys()) | set(coco_result.keys()):
        new_result[rouge] = {}
        if rouge in original_result:
            new_result[rouge].update(original_result[rouge])
        if rouge in coco_result:
            new_result[rouge].update(coco_result[rouge])

    return new_result


def calculate_f1score(candidate, reference, normalize):
    if normalize:
        candidate = normalize_word(candidate)
        reference = normalize_word(reference)

    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    word_set = set()
    for word in candidate_words:
        word_set.add(word)
    for word in reference_words:
        word_set.add(word)

    tp = 0
    fp = 0
    fn = 0
    for word in word_set:
        if word in candidate_words and word in reference_words:
            tp += candidate_words[word]
        elif word in candidate_words and word not in reference_words:
            fp += candidate_words[word]
        elif word not in candidate_words and word in reference_words:
            fn += reference_words[word]

    if len(candidate_words) == 0:
        return 0
    elif len(reference_words) == 0:
        return 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if tp == 0:
            return 0
        else:
            return 2 * precision * recall / (precision + recall)


def calculate_f1(data):
    f1_no_norm_list = []
    f1_norm_list = []
    
    for ann in data:
        f1_norm_list.append(calculate_f1score(ann["answer_pred"], ann["answer_gt"], normalize=True))
        f1_no_norm_list.append(calculate_f1score(ann["answer_pred"], ann["answer_gt"], normalize=False))

    f1_norm_list = np.array(f1_norm_list).mean()
    f1_no_norm_list = np.array(f1_no_norm_list).mean()

    return {
        'no_norm': f1_no_norm_list,
        'norm': f1_norm_list,
    }


def calculate_vqa_acc(data):
    acc_scores = []
    for ann in data:
        answer_pred = normalize_word(ann["answer_pred"])
        answer_gt = normalize_word(ann["answer_gt"])
        acc = 1 if answer_gt in answer_pred else 0
        acc_scores.append(acc)
    acc_scores = np.array(acc_scores).mean()

    return acc_scores


def calculate_equal_acc(data):
    acc_scores = []
    for ann in data:
        answer_pred = normalize_word(ann["answer_pred"])
        answer_gt = normalize_word(ann["answer_gt"])
        acc = 1 if answer_gt == answer_pred else 0
        acc_scores.append(acc)
    acc_scores = np.array(acc_scores).mean()

    return acc_scores


def evaluate_cls(pred_json,gt_json):
    acc_scores = []
    for pred, gt in zip(pred_json, gt_json):
        answer_pred = normalize_word(pred)
        answer_gt = normalize_word(gt["conversations"][1]["value"])
        acc = 1 if answer_gt == answer_pred else 0
        acc_scores.append(acc)
    acc_scores = np.array(acc_scores).mean()
    return {"accuracy": acc_scores}


def evaluate_caption(pred_json,gt_json):    
    uni_med_predict_filtered = [ann for ann in uni_med_predict if ann['answer_gt'] and ann['answer_pred']]

    bleu_scores = calculate_bleu(uni_med_predict_filtered)
    meteor_scores = calculate_meteor(uni_med_predict_filtered)
    rouge_scores = calculate_rouge(uni_med_predict_filtered)
    f1_scores = calculate_f1(uni_med_predict_filtered)

    return {
        "bleu_scores": bleu_scores,
        "meteor_scores": meteor_scores,
        "rouge_scores": rouge_scores,
        "f1_score": f1_scores,
    }


############################cls
# medmnist_organs
pred_path = os.path.join(path, 'medmnist_organs.json')
gt_path = os.path.join(gt_ori, 'medmnist_organs.json')

with open(pred_path,'r') as f1:
    data1 = json.load(f1)

with open(gt_path,'r') as f2:
    data2 = json.load(f2)

data_all = []
for pred, gt in zip(data1,data2):
    data = {
        "answer_pred": pred,
        "answer_gt":gt["conversations"][1]["value"]
    }
    data_all.append(data)

metrics = calculate_equal_acc(data_all)
metric_save_path = os.path.join(metric, 'medmnist_organs.json')
with open(metric_save_path, 'w') as f:
    json.dump(metrics, f, sort_keys=True)
print("save the metrics to {}".format(metric_save_path))
print("metrics: {}".format(metrics))
results['medmnist_organs'] = metrics


# medmnist_derma
pred_path = os.path.join(path, 'medmnist_derma.json')
gt_path = os.path.join(gt_ori, 'medmnist_derma.json')

with open(pred_path,'r') as f1:
    data1 = json.load(f1)

with open(gt_path,'r') as f2:
    data2 = json.load(f2)

data_all = []
for pred, gt in zip(data1,data2):
    data = {
        "answer_pred": pred,
        "answer_gt":gt["conversations"][1]["value"]
    }
    data_all.append(data)

metrics = calculate_equal_acc(data_all)
metric_save_path = os.path.join(metric, 'medmnist_derma.json')
with open(metric_save_path, 'w') as f:
    json.dump(metrics, f, sort_keys=True)
print("save the metrics to {}".format(metric_save_path))
print("metrics: {}".format(metrics))
results['medmnist_derma'] = metrics


############################caption
# mimic
pred_path = os.path.join(path, 'mimic.json')
gt_path = os.path.join(gt_ori, 'mimic.json')

with open(pred_path,'r') as f1:
    data1 = json.load(f1)

with open(gt_path,'r') as f2:
    data2 = json.load(f2)

data_all = []
for pred, gt in zip(data1,data2):
    data = {
        "answer_pred": pred,
        "answer_gt": gt["conversations"][1]["value"]
    }
    data_all.append(data)

def evaluate_all(uni_med_predict):    
    
    uni_med_predict_filtered = [ann for ann in uni_med_predict if ann['answer_gt'] and ann['answer_pred']]

    bleu_scores = calculate_bleu(uni_med_predict_filtered)
    meteor_scores = calculate_meteor(uni_med_predict_filtered)
    rouge_scores = calculate_rouge(uni_med_predict_filtered)
    f1_scores = calculate_f1(uni_med_predict_filtered)

    return {
        "bleu_scores": bleu_scores,
        "meteor_scores": meteor_scores,
        "rouge_scores": rouge_scores,
        "f1_score": f1_scores,
    }

metrics = evaluate_all(data_all)
metric_save_path = os.path.join(metric, 'mimic.json')
with open(metric_save_path, 'w') as f:
    json.dump(metrics, f, sort_keys=True)
print("save the metrics to {}".format(metric_save_path))
print("metrics: {}".format(metrics))
results['mimic'] = metrics["bleu_scores"]["bleu1"]["coco_no_norm"]


# medpix
pred_path = os.path.join(path, 'medpix.json')
gt_path = os.path.join(gt_ori, 'medpix.json')

with open(pred_path,'r') as f1:
    data1 = json.load(f1)

with open(gt_path,'r') as f2:
    data2 = json.load(f2)

data_all = []
for pred, gt in zip(data1,data2):
    data = {
        "answer_pred": pred,
        "answer_gt": gt["conversations"][1]["value"]
    }
    data_all.append(data)

metrics = evaluate_all(data_all)
metric_save_path = os.path.join(metric, 'medpix.json')
with open(metric_save_path, 'w') as f:
    json.dump(metrics, f, sort_keys=True)
print("save the metrics to {}".format(metric_save_path))
print("metrics: {}".format(metrics))
results['medpix'] = metrics["bleu_scores"]["bleu1"]["coco_no_norm"]


############################vqa
# slakevqa
pred_path = os.path.join(path, 'slakevqa.json')
gt_path = os.path.join(gt_ori, 'slakevqa.json')

with open(pred_path,'r') as f1:
    data1 = json.load(f1)

with open(gt_path,'r') as f2:
    data2 = json.load(f2)

data_all = []
for pred, gt in zip(data1,data2):
    data = {
        "answer_pred": pred,
        "answer_gt": gt["conversations"][1]["value"]
        
    }
    data_all.append(data)

def evaluate_vqa_all(uni_med_predict):    
    all_bleu_scores = calculate_bleu(uni_med_predict)
    all_f1_scores = calculate_f1(uni_med_predict)

    return {
        "all": {
            "bleu_scores": all_bleu_scores,
            "f1_score": all_f1_scores,
        }
    }

metrics = evaluate_vqa_all(data_all)
metric_save_path = os.path.join(metric, 'slakevqa.json')
with open(metric_save_path, 'w') as f:
    json.dump(metrics, f, sort_keys=True)
print("save the metrics to {}".format(metric_save_path))
print("metrics: {}".format(metrics))
results['slakevqa'] = metrics["all"]["bleu_scores"]["bleu1"]["nltk_norm"]


# pathvqa
pred_path = os.path.join(path, 'pathvqa.json')
gt_path = os.path.join(gt_ori, 'pathvqa.json')

with open(pred_path,'r') as f1:
    data1 = json.load(f1)

with open(gt_path,'r') as f2:
    data2 = json.load(f2)

data_all = []
for pred, gt in zip(data1,data2):
    data = {
        "answer_pred": pred,
        "answer_gt": gt["conversations"][1]["value"]
        
    }
    data_all.append(data)

metrics = evaluate_vqa_all(data_all)
metric_save_path = os.path.join(metric, 'pathvqa.json')
with open(metric_save_path, 'w') as f:
    json.dump(metrics, f, sort_keys=True)
print("save the metrics to {}".format(metric_save_path))
print("metrics: {}".format(metrics))
results['pathvqa'] = metrics["all"]["bleu_scores"]["bleu1"]["nltk_norm"]


############################identify
# slake_identify
pred_path = os.path.join(path, 'slake_identify.json')
gt_path = os.path.join(gt_ori, 'slake_identify.json')

with open(pred_path,'r') as f1:
    data1 = json.load(f1)

with open(gt_path,'r') as f2:
    data2 = json.load(f2)

data_all = []
for pred, gt in zip(data1,data2):
    data = {
        "answer_pred": pred,
        "answer_gt": gt["conversations"][1]["value"]
        
    }
    data_all.append(data)

def evaluate_iden(uni_med_predict):

    bleu_scores = calculate_bleu(uni_med_predict)
    f1_scores = calculate_f1(uni_med_predict)
    acc_scores = calculate_equal_acc(uni_med_predict)

    return {
        "bleu_scores": bleu_scores,
        "f1_score": f1_scores,
        "acc_score": acc_scores,
    }

metrics = evaluate_iden(data_all)
metric_save_path = os.path.join(metric, 'slake_identify.json')
with open(metric_save_path, 'w') as f:
    json.dump(metrics, f, sort_keys=True)
print("save the metrics to {}".format(metric_save_path))
print("metrics: {}".format(metrics))
results['slake_identify'] = metrics["bleu_scores"]["bleu1"]["nltk_norm"]


# samed_identify
pred_path = os.path.join(path, 'samed_identify.json')
gt_path = os.path.join(gt_ori, 'samed_identify.json')

with open(pred_path,'r') as f1:
    data1 = json.load(f1)

with open(gt_path,'r') as f2:
    data2 = json.load(f2)

data_all = []
for pred, gt in zip(data1,data2):
    data = {
        "answer_pred": pred,
        "answer_gt": gt["conversations"][1]["value"]
        
    }
    data_all.append(data)

def evaluate_iden(uni_med_predict):
    
    bleu_scores = calculate_bleu(uni_med_predict)
    f1_scores = calculate_f1(uni_med_predict)
    acc_scores = calculate_equal_acc(uni_med_predict)

    return {
        "bleu_scores": bleu_scores,
        "f1_score": f1_scores,
        "acc_score": acc_scores,
    }

metrics = evaluate_iden(data_all)
metric_save_path = os.path.join(metric, 'samed_identify.json')
with open(metric_save_path, 'w') as f:
    json.dump(metrics, f, sort_keys=True)
print("save the metrics to {}".format(metric_save_path))
print("metrics: {}".format(metrics))
results['samed_identify'] = metrics["bleu_scores"]["bleu1"]["nltk_norm"]


############################refer
# samed_refer
pred_path = os.path.join(path, 'samed_refer.json')
gt_path = os.path.join(gt_ori, 'samed_refer.json')

with open(pred_path,'r') as f1:
    data1 = json.load(f1)

with open(gt_path,'r') as f2:
    data2 = json.load(f2)

iou_scores = []
res = 100

for pred, gt in zip(data1,data2):
    answer = pred
    bbox = gt["conversations"][1]["value"]
    image_size = gt["image_size"]

    gt_bbox = [0,0,0,0]
    gt_bbox[0] = bbox[0]
    gt_bbox[1] = bbox[1]
    gt_bbox[2] = bbox[0] + bbox[2]
    gt_bbox[3] = bbox[1] + bbox[3]

    answer = answer.replace("<unk>","").replace(" ","").strip()
    pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'

    if re.match(pattern, answer):
        integers = re.findall(r'\d+', answer)
        pred_bbox = [int(num) for num in integers][:4]
        width = image_size[0]
        height = image_size[1]
        pred_bbox[0] = pred_bbox[0] / res * width
        pred_bbox[1] = pred_bbox[1] / res * height
        pred_bbox[2] = pred_bbox[2] / res * width
        pred_bbox[3] = pred_bbox[3] / res * height

        iou_score = computeIoU(pred_bbox, gt_bbox)
        iou_scores.append(iou_score)
    else:
        iou_scores.append(0)

iou_scores = np.array(iou_scores)
metrics = {'miou': iou_scores.mean(), 'acc': (iou_scores>0.5).mean()}
metric_save_path = os.path.join(metric, 'samed_refer.json')
with open(metric_save_path, 'w') as f:
    json.dump(metrics, f, sort_keys=True)
print("save the metrics to {}".format(metric_save_path))
print("metrics: {}".format(metrics))
results['samed_refer'] = metrics["miou"]


# slake_refer
pred_path = os.path.join(path, 'slake_refer.json')
gt_path = os.path.join(gt_ori, 'slake_refer.json')

with open(pred_path,'r') as f1:
    data1 = json.load(f1)

with open(gt_path,'r') as f2:
    data2 = json.load(f2)

iou_scores = []
res = 100

for pred, gt in zip(data1,data2):
    answer = pred
    bbox = gt["conversations"][1]["value"]
    image_size = gt["image_size"]

    gt_bbox = [0,0,0,0]
    gt_bbox[0] = bbox[0]
    gt_bbox[1] = bbox[1]
    gt_bbox[2] = bbox[0] + bbox[2]
    gt_bbox[3] = bbox[1] + bbox[3]

    answer = answer.replace("<unk>","").replace(" ","").strip()
    pattern = r'\{<\d{1,3}><\d{1,3}><\d{1,3}><\d{1,3}>\}'

    if re.match(pattern, answer):
        integers = re.findall(r'\d+', answer)
        pred_bbox = [int(num) for num in integers][:4]
        width = image_size[0]
        height = image_size[1]
        pred_bbox[0] = pred_bbox[0] / res * width
        pred_bbox[1] = pred_bbox[1] / res * height
        pred_bbox[2] = pred_bbox[2] / res * width
        pred_bbox[3] = pred_bbox[3] / res * height

        iou_score = computeIoU(pred_bbox, gt_bbox)
        iou_scores.append(iou_score)
    else:
        iou_scores.append(0)

iou_scores = np.array(iou_scores)
metrics = {'miou': iou_scores.mean(), 'acc': (iou_scores>0.5).mean()}
metric_save_path = os.path.join(metric, 'slake_refer.json')
with open(metric_save_path, 'w') as f:
    json.dump(metrics, f, sort_keys=True)
print("save the metrics to {}".format(metric_save_path))
print("metrics: {}".format(metrics))
results['slake_refer'] = metrics["miou"]


# save results
df = pd.DataFrame([results])
cols = ["medmnist_derma", "medmnist_organs", "medpix", "mimic", "pathvqa", "samed_identify", "samed_refer", "slake_identify", "slake_refer", "slakevqa"]
df = df[cols]
df.to_csv(os.path.join(metric, 'results.csv'), index=False)
