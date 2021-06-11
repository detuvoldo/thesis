#!/usr/bin/env python
# coding: utf-8
# %%

# # Pseudo Decomposition Preparation

# %%


import json
from tqdm import tqdm
import argparse
import os
from os.path import join


# %%
import nltk
from nltk import word_tokenize
nltk.download('punkt')


# %%
parser = argparse.ArgumentParser(description="Pseudo Decomposition Preparation")

# %%
parser.add_argument("--data_folder",
                   type=str,
                   default="data",
                   help="Data folder")
parser.add_argument("--output_folder",
                   type=str,
                   default="data",
                   help="Output folder")

# %%
hotpot_train = "hotpot_train_v1.1.json"
hotpot_dev = "hotpot_dev_distractor_v1.json"

hotpot_train_out = "out_train.json"
hotpot_dev_out = "out_dev.json"



# %%

def bridge_ent(sample):
    bridge_ent = ''
    case = 0
    
    paraA = " ".join(sample['gold_para_1'])
    paraB = " ".join(sample['gold_para_2'])
    entA = sample['title_ent_1']
    entB = sample['title_ent_2']
    question = sample['question']
    answer = sample['answer']
    
    # Case 1
    if entA.lower() in paraB.lower() and entB.lower() not in paraA.lower():
        case = 1
        return entA, case
    if entA.lower() not in paraB.lower() and entB.lower() in paraA.lower():
        case = 1
        return entB, case
    
    # Case 2
    if entA.lower() not in paraB.lower() and entB.lower() not in paraA.lower():
        case = 2
        entA_tokens = word_tokenize(entA)
        entB_tokens = word_tokenize(entB)
        countA = 0
        countB = 0

        for token in entA_tokens:
            if token in paraB:
                countA += 1
        for token in entB_tokens:
            if token in paraA:
                countB += 1

        if countA / float(len(entA_tokens)) > countB / float(len(entB_tokens)):
            return entA, case
        if countA / float(len(entA_tokens)) < countB / float(len(entB_tokens)):
            return entB, case
        
    # Case 3
    if entA.lower() in paraB.lower() and entB.lower() in paraA.lower():
        case = 3
        
        if entA.lower() in question.lower() and entA.lower() in answer.lower():
            return entB, case
        if entB.lower() in question.lower() and entB.lower() in answer.lower():
            return entA, case
    return bridge_ent, case


# %%
if __name__ == '__main__':
    args = parser.parse_args()
    
    with open(join(args.data_folder, hotpot_train), 'r') as hotpot_train_f:
        hotpot_train_data = json.load(hotpot_train_f)

    bridge_hotpotqa_train_data = []
    for sample in tqdm(hotpot_train_data):
        if sample['type'] == 'bridge':
            tmp = {
                "_id": sample['_id'],
                "question": sample['question'],
                "answer": sample['answer'],
                "gold_para_1": [],
                "gold_para_2": [],
                "title_ent_1": '',
                "title_ent_2": '',
                "supporting_facts_1": [],
                "supporting_facts_2": [],
                "bridge_ent": ''            
            }
            gold_title_ents = []
            gold_title_ents.extend(ent[0] for ent in sample['supporting_facts'] if ent[0] not in gold_title_ents)
            tmp['title_ent_1'] = gold_title_ents[0]
            tmp['title_ent_2'] = gold_title_ents[1]

            for fact in sample['supporting_facts']:
                for para in sample['context']:
                    if tmp['title_ent_1'] == para[0] == fact[0]:
                        try:
                            tmp["supporting_facts_1"].append(fact[1])
                            tmp['gold_para_1'].extend(para[1])
                        except Exception:
                            continue
                    if tmp['title_ent_2'] == para[0] == fact[0]:
                        try:
                            tmp["supporting_facts_2"].append(fact[1])
                            tmp['gold_para_2'].extend(para[1])
                        except Exception:
                            continue
            bridge_hotpotqa_train_data.append(tmp)
            
    for sample in tqdm(bridge_hotpotqa_train_data):
        sample['bridge_ent_case'] = 0
        sample['bridge_ent'], sample['bridge_ent_case'] = bridge_ent(sample)
        
    only_bridges = []

    for sample in bridge_hotpotqa_train_data:
        if sample['bridge_ent'].lower() in sample['answer'].lower():
            continue
        elif sample['bridge_ent'].lower() in sample['question'].lower():
            continue
        else:
            only_bridges.append(sample)
            
    try:
        os.mkdir(args.output_folder)
        #print(1)
        with open(join(args.output_folder, hotpot_train_out), 'w') as br_file:
            json.dump(only_bridges, br_file, indent=4)
            
    except Exception:
        with open(join(args.output_folder, hotpot_train_out), 'w') as br_file:
            json.dump(only_bridges, br_file, indent=4)
        
    ################################################
    
    with open(join(args.data_folder, hotpot_dev), 'r') as hotpot_dev_f:
        hotpot_dev_data = json.load(hotpot_dev_f)

    bridge_hotpotqa_dev_data = []
    for sample in tqdm(hotpot_dev_data):
        if sample['type'] == 'bridge':
            tmp = {
                "_id": sample['_id'],
                "question": sample['question'],
                "answer": sample['answer'],
                "gold_para_1": [],
                "gold_para_2": [],
                "title_ent_1": '',
                "title_ent_2": '',
                "supporting_facts_1": [],
                "supporting_facts_2": [],
                "bridge_ent": ''            
            }
            gold_title_ents = []
            gold_title_ents.extend(ent[0] for ent in sample['supporting_facts'] if ent[0] not in gold_title_ents)
            tmp['title_ent_1'] = gold_title_ents[0]
            tmp['title_ent_2'] = gold_title_ents[1]

            for fact in sample['supporting_facts']:
                for para in sample['context']:
                    if tmp['title_ent_1'] == para[0] == fact[0]:
                        try:
                            tmp["supporting_facts_1"].append(fact[1])
                            tmp['gold_para_1'].extend(para[1])
                        except Exception:
                            continue
                    if tmp['title_ent_2'] == para[0] == fact[0]:
                        try:
                            tmp["supporting_facts_2"].append(fact[1])
                            tmp['gold_para_2'].extend(para[1])
                        except Exception:
                            continue
            bridge_hotpotqa_dev_data.append(tmp)
            
    for sample in tqdm(bridge_hotpotqa_dev_data):
        sample['bridge_ent_case'] = 0
        sample['bridge_ent'], sample['bridge_ent_case'] = bridge_ent(sample)
        
    only_dev_bridges = []

    for sample in bridge_hotpotqa_dev_data:
        if sample['bridge_ent'].lower() in sample['answer'].lower():
            continue
        elif sample['bridge_ent'].lower() in sample['question'].lower():
            continue
        else:
            only_dev_bridges.append(sample)
            
    try:
        os.mkdir(args.output_folder)
        #print(1)
        with open(join(args.output_folder, hotpot_dev_out), 'w') as dev_br_file:
            json.dump(only_dev_bridges, dev_br_file, indent=4)
            
    except Exception:
        with open(join(args.output_folder, hotpot_dev_out), 'w') as dev_br_file:
            json.dump(only_dev_bridges, dev_br_file, indent=4)

# %%
# ### Rules of selecting brigde entities from title entities
#      1. If the title entity EA of paragraph A occurs in the other paragraph B, 
#      while the title entity EB of B DOES NOT occur in A, then EA is recognized as the bridge entity.
#      
#      2. Second, if neither EA nor EB is contained in the other paragraph, 
#      then the title entity with MORE OVERLAPPING TEXT with the other paragraph is chosen as the bridge entity
#      (since sometimes the alias of the Wikipedia title is used in the paragraph).
#      
#      3.  if both EA and EB appear in the other paragraph, then the title entity which DOES NOT appear
#      in both the question and the answer is chosen as the bridge entity
#      
#      The bridge entity is set to be unidentified if none or both of the title entities satisfy at least one of the requirements
