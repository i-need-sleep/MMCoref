import os
import json
import torch

def get_output(json_list, weights, test_path='..data/simmc2_dials_dstc10_teststd_public.json'):

    inf_list = []
    for json_file in json_list:
        with open(json_file, 'r', encoding='utf-8') as f:
            inf_list.append(json.load(f))

    with open(test_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
            
    for dial in data['dialogue_data']:
        dial_mentions = []
        dial_idx = dial['dialogue_idx']
        for round_idx, round in enumerate(dial['dialogue']):
            try:
                round['transcript_annotated']['act_attributes']['objects'] = []
            except:
                round['transcript_annotated'] = {}
                round['transcript_annotated']['act_attributes'] = {}
                round['transcript_annotated']['act_attributes']['objects'] = []
            try:
                for obj_idx in inf_list[0][str(dial_idx)][str(round_idx)].keys():
                    pred = 0
                    for inf_idx, inf_dict in enumerate(inf_list):
                        pred += inf_dict[str(dial_idx)][str(round_idx)][obj_idx] * weights[inf_idx] / 5
                        # print(inf_dict[str(dial_idx)][str(round_idx)][obj_idx])
                    if pred > -0.15389795803714482:
                        round['transcript_annotated']['act_attributes']['objects'].append(int(obj_idx))
                        if int(obj_idx) not in dial_mentions:
                            dial_mentions.append(int(obj_idx))
            except:
                assert(round['disambiguation_label'] == 1)
        dial['mentioned_object_ids'] = dial_mentions
    
    with open(f'./output/output.json', 'w', encoding='utf-8') as out_file:
        json.dump(data, out_file)

if __name__ == '__main__':
    json_list = [
        './inference/base_teststd_obj_logits.json',
        './inference/attnbias_rcnn_teststd_obj_logits.json',
        './inference/KBid_teststd_obj_logits.json',
        './inference/sceneseg_teststd_obj_logits.json',
        './inference/attnbias_rcnn_SBERT_graph_teststd_obj_logits.json',
    ]
    weights = [0.8936481587327065, 0.8107087489638614, 0.6766842033721806, 0.5994568035325222, 0.640284116757462]
    get_output(json_list, weights, test_path='../data/simmc2_dials_dstc10_teststd_public.json')