#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import json
from tqdm import tqdm
import os
from os.path import join
import argparse


# %%


from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') 


# %%


parser = argparse.ArgumentParser(description="Prepare Data for Training QD Model")
parser.add_argument("--data_folder",
                   type=str,
                   default="out",
                   help="Data Folder")

parser.add_argument("--output_folder",
                   type=str,
                   default="out",
                   help="Output folder")


# %%


# Replace bridge entity in the 2nd sub-question by [UNK] token

def merge_tokens(token_list):
    cur_tok = ""
    res = []
    
    for tok in token_list:
        if tok[:2] == "##":
            cur_tok = cur_tok + tok[2:]
            res[-1] = cur_tok
        else:
            cur_tok = tok
            res.append(tok)
    return " ".join(res)

def replace_bridge_ent(ent, sent):
    ent_tokens = tokenizer.tokenize(ent)
    sent_tokens = tokenizer.tokenize(sent)
    ent_start, ent_end = -1, -1
    
    for i in range(len(sent_tokens)):
        if sent_tokens[i:i+len(ent_tokens)] == ent_tokens:
            ent_start = i
            ent_end = i + len(ent_tokens)
        else:
            ent_len = len(ent_tokens) - 1
            while ent_len > 0:
                if sent_tokens[i:i+ent_len] == ent_tokens[:ent_end]:
                    ent_start = i
                    ent_end = i + ent_len
                    break
                ent_len -= 1
                
    new_sent_tokens = sent_tokens[:ent_start] + ["[br**@@]"] + sent_tokens[ent_end:]
    return merge_tokens(new_sent_tokens)    


# %%


if __name__ == "__main__":
    args = parser.parse_args()
    
    with open(join(args.data_folder, "train_decomp.json"), 'r') as train_f:
        train_data = json.load(train_f)
        
    questions = []
    sub_questions = []
    sub_1s = []
    sub_2s = []
    question_ids = []
    
    train_data = [sample for sample in train_data if len(sample['sub_questions']) > 1]
    
    for sample in tqdm(train_data):
        #print(sample)
        question = sample['multihop_question']
        sub_q_1 = sample['sub_questions'][0]['subquestion_1']
        sub_q_2 = sample['sub_questions'][1]['subquestion_2']
        bridge_ent = sample['sub_questions'][0]['bridge_ent'].lower()
        question_ids.append(sample['_id'])

        if bridge_ent in sub_q_2:
            sub_q_2 = sub_q_2.replace(bridge_ent, "[br**@@]")

        else:
            sub_q_2 = replace_bridge_ent(bridge_ent, sub_q_2)

        sub_qs = sub_q_1 + " " + sub_q_2
        #print(sub_qs)
        questions.append(question)
        sub_questions.append(sub_qs)
        sub_1s.append(sub_q_1)
        sub_2s.append(sub_q_2)

    train_size = int(len(questions) * 0.8)
    assert len(questions) == len(sub_questions)
    
    qd_train_questions = questions[:train_size]
    qd_train_subquestions = sub_questions[:train_size]
    qd_train_sub_1s = sub_1s[:train_size]
    qd_train_sub_2s = sub_2s[:train_size]
    qd_train_ids = question_ids[:train_size]

    qd_valid_questions = questions[train_size:]
    qd_valid_subquestions = sub_questions[train_size:]
    qd_valid_sub_1s = sub_1s[train_size:]
    qd_valid_sub_2s = sub_2s[train_size:]
    
    with open(join(args.data_folder, 'dev_decomp.json'), 'r') as dev_f:
        dev_data = json.load(dev_f)

    dev_questions = []
    dev_questions_tok = []
    dev_sub_questions = []
    dev_sub_questions_tok = []
    dev_sub1s = []
    dev_sub2s = []
    valid_ids = []

    for sample in tqdm(dev_data):
        question = sample['question']
        bridge_ent = sample['bridge_entity'].lower()
        idx = sample['_id']

        #if contain_br_ent(question, bridge_ent):
        sub_q_1 = sample['sub_question_1']
        sub_q_2 = sample['sub_question_2']


        if bridge_ent in sub_q_2:
            sub_q_2 = sub_q_2.replace(bridge_ent, "[br**@@]")

        else:
            sub_q_2 = replace_bridge_ent(bridge_ent, sub_q_2)

        sub_qs = sub_q_1 + " " + sub_q_2
        #print(sub_qs)
        dev_questions.append(question)
        dev_sub_questions.append(sub_qs)
        dev_questions_tok.append(question.lower())
        dev_sub_questions_tok.append(sub_qs.lower())
        dev_sub1s.append(sub_q_1)
        dev_sub2s.append(sub_q_2)
        valid_ids.append(idx)
        
        
    with open(join(args.output_folder, "train.mh"), "w") as train_mh:
        for q in qd_train_questions:
            train_mh.write("%s\n" %q)
    with open(join(args.output_folder, "train.sh"), "w") as train_sh:
        for q in qd_train_subquestions:
            train_sh.write("%s\n" %q)

    with open(join(args.output_folder, "train.sh1"), "w") as train_sh1:
        for q in qd_train_sub_1s:
            train_sh1.write("%s\n" %q)

    with open(join(args.output_folder, "train.sh2"), "w") as train_sh2:
        for q in qd_train_sub_2s:
            train_sh2.write("%s\n" %q)

    with open(join(args.output_folder, "train.qids.txt"), "w") as train_qids:
        for q in qd_train_ids:
            train_qids.write("%s\n" %q)

    with open(join(args.output_folder, "test.mh"), "w") as test_mh:
        for q in qd_valid_questions:
            test_mh.write("%s\n" %q)

    with open(join(args.output_folder, "test.sh"), "w") as test_sh:
        for q in qd_valid_subquestions:
            test_sh.write("%s\n" %q)

    with open(join(args.output_folder, "test.sh1"), "w") as test_sh1:
        for q in qd_valid_sub_1s:
            test_sh1.write("%s\n" %q)        

    with open(join(args.output_folder, "test.sh2"), "w") as test_sh2:
        for q in qd_valid_sub_2s:
            test_sh2.write("%s\n" %q)
            
    with open(join(args.output_folder, "valid.mh"), "w") as valid_mh:
        for q in dev_questions:
            valid_mh.write("%s\n" %q)

    with open(join(args.output_folder, "valid.sh"), "w") as valid_sh:
        for q in dev_sub_questions:
            valid_sh.write("%s\n" %q)

    with open(join(args.output_folder, "valid.sh1"), "w") as valid_sh1:
        for q in dev_sub1s:
            valid_sh1.write("%s\n" %q)

    with open(join(args.output_folder, "valid.sh2"), "w") as valid_sh2:
        for q in dev_sub2s:
            valid_sh2.write("%s\n" %q)

    with open(join(args.output_folder, "valid.qids.txt"), 'w') as valid_qid:
        for idx in valid_ids:
            valid_qid.write("%s\n" %idx)

    with open(join(args.output_folder, "valid.mh.tok"), 'w') as valid_mh_tok:
        for q in dev_questions_tok:
            valid_mh_tok.write('%s\n' %q)

    with open(join(args.output_folder, "valid.sh.tok"), 'w') as valid_sh_tok:
        for q in dev_sub_questions_tok:
            valid_sh_tok.write('%s\n' %q)


# %%




