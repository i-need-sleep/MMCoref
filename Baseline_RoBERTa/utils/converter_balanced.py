import json
from numpy.lib.twodim_base import mask_indices
import tqdm

import numpy as np

DATA_ROOT = "../../data"
OUT_ROOT = "../processed"
OBJ_TOKEN = '<mask>'

ALLOWED_KB = ['color', 'type']
MAX_N_OBJECTS = 35
NEGATIVE_RATIO = 1

ref_error = 0

def dial_KB_to_string(split = "train", dial_mem = 1, KB_hist = False, KB_cand = True):

    # Flatten dialogue json into strings with relavant KB entries incorporated
    # One line for each round

    dial_path = f"{DATA_ROOT}/simmc2_dials_dstc10_{split}.json"
    scene_root = f"{DATA_ROOT}/simmc2_scene_jsons_dstc10_public/public"
    KB_fashion_path = f"{DATA_ROOT}/fashion_prefab_metadata_all.json"
    KB_furniture_path = f"{DATA_ROOT}/furniture_prefab_metadata_all.json"
    
    max_len = 0
    
    with open(dial_path, 'r') as dial_file:
        dials = json.load(dial_file)['dialogue_data']

    with open(KB_fashion_path, 'r') as fashion_file:
        KB_fash = json.load(fashion_file)

    with open(KB_furniture_path, 'r') as furniture_file:
        KB_fur = json.load(furniture_file)

    out = ''

    for dial_idx, dial in enumerate(tqdm.tqdm(dials)):
        dial_hist = dial['dialogue']
        domain = dial['domain']
        scene_ids = dial['scene_ids']
        
        for round_idx, cur_round in enumerate(dial_hist):
            round_out = ''

            for r in range(max(0, round_idx - dial_mem), round_idx):
                round = dial_hist[r]
                round_out += f'User : {round["transcript"]} '
                
                round_scene_key = '0'
                for scene_key in scene_ids:
                    if int(scene_key) <= int(round_idx) and int(scene_key) > int(round_scene_key):
                        round_scene_key = scene_key
                round_scene_id = scene_ids[round_scene_key]


                if not KB_hist:
                    for idx, obj in enumerate(round["transcript_annotated"]["act_attributes"]["objects"]):
                        round["transcript_annotated"]["act_attributes"]["objects"][idx] = str(obj)
                    round_out += f'Objects : {" ".join(round["transcript_annotated"]["act_attributes"]["objects"])} '
                else:
                    if domain == 'fashion':
                        ref_KB = KB_fash
                    else:
                        ref_KB = KB_fur
                    KB_out = KB_retrive(round["transcript_annotated"]["act_attributes"]["objects"], round_scene_id, ref_KB)
                    if KB_out == 'break':
                        break
                    round_out += KB_out
                round_out += f'System : {round["system_transcript"]} '
            
            else:
                round_out += f'User : {cur_round["transcript"]} '
                
                round_scene_key = '0'
                for scene_key in scene_ids:
                    if int(scene_key) <= int(round_idx) and int(scene_key) > int(round_scene_key):
                        round_scene_key = scene_key
                round_scene_id = scene_ids[round_scene_key]

                if domain == 'fashion':
                    ref_KB = KB_fash
                else:
                    ref_KB = KB_fur

                if KB_cand:
                    KB_out, mask_idx = KB_retrive_cand(cur_round["transcript_annotated"]["act_attributes"]["objects"],round_scene_id, ref_KB)
                    if KB_out == 'break':
                        continue
                    round_out += KB_out
                    round_out += f'=> Predict : {" ".join(mask_idx)}\n'

                out += round_out.replace('  ',' ')
                if len(round_out) > max_len:
                    max_len = len(round_out)
                continue
            break
        
        # if dial_idx > 5:
        #     break

    with open(f'{OUT_ROOT}/balanced_{split}.txt', 'w', encoding='utf-8') as out_file:
        out_file.write(out)

    print(f'#Idx_error: {ref_error}')
    return

def KB_retrive(obj_indices, scene_id, KB):
    scene_root = f"{DATA_ROOT}/simmc2_scene_jsons_dstc10_public/public"
    scene_path = f'{scene_root}/{scene_id}_scene.json'

    with open(scene_path, 'r') as scene_file:
        scene_data = json.load(scene_file)['scenes'][0]['objects']

    out = 'Objects : '
    for obj_idx in obj_indices:
        obj_idx = int(obj_idx)
        try:
            position_str = f"Position : {' '.join(['{:.2f}'.format(pos) for pos in scene_data[obj_idx]['position']])}, "
        except:
            global ref_error
            ref_error += 1
            return "break"
            
        
        prefab = KB[scene_data[obj_idx]['prefab_path']]
        KB_str = ''
        for key, val in prefab.items():
            if key not in ALLOWED_KB:
                continue
            if type(val) == type([]):
                val = ' '.join(val)
            KB_str += f'{key} : {val} '
        
        obj_string = f'{str(obj_idx)} : {position_str}{KB_str}, '
        out += obj_string

    return out

def KB_retrive_idx(obj_indices, scene_id, KB):
    global ref_error
    scene_root = f"{DATA_ROOT}/simmc2_scene_jsons_dstc10_public/public"
    scene_path = f'{scene_root}/{scene_id}_scene.json'

    with open(scene_path, 'r') as scene_file:
        scene_data = json.load(scene_file)['scenes'][0]['objects']

    if len(scene_data) > MAX_N_OBJECTS:
        ref_error += 1
        return 'break'

    out = ''
    for obj_idx in obj_indices:
        obj_idx = int(obj_idx)
        try:
            position_str = f"Position : {' '.join(['{:.2f}'.format(pos) for pos in scene_data[obj_idx]['position']])}, "
        except:
            ref_error += 1
            return "break"

        out += f'{obj_idx} '

    return out

def KB_retrive_cand(obj_indices, scene_id, KB):
    scene_root = f"{DATA_ROOT}/simmc2_scene_jsons_dstc10_public/public"
    scene_path = f'{scene_root}/{scene_id}_scene.json'

    with open(scene_path, 'r') as scene_file:
        scene_data = json.load(scene_file)['scenes'][0]['objects']

    out = 'Candidates : '
    mask_idx = []
    negative_prob = max(len(obj_indices), 1) * NEGATIVE_RATIO /(len(scene_data) - len(obj_indices))
    if len(obj_indices) == 0:
        return 'break', 'break'
    ctr = 0
    for obj_idx in range(len(scene_data)):
        if not obj_idx in obj_indices:
            if np.random.random() > negative_prob and not (ctr == 0 and obj_idx == len(scene_data)-1):
                continue
        else:
            mask_idx.append(str(ctr))
        ctr += 1
        position_str = f"Position : {' '.join(['{:.2f}'.format(pos) for pos in scene_data[obj_idx]['position']])}, "
            
        prefab = KB[scene_data[obj_idx]['prefab_path']]
        KB_str = ''
        for key, val in prefab.items():
            if key not in ALLOWED_KB:
                continue
            if type(val) == type([]):
                val = ' '.join(val)
            # KB_str += f'{key} : {val} , '
            KB_str += f'{val} , '
        
        # obj_string = f'</s> {str(obj_idx)} : {position_str}{KB_str} '
        obj_string = f'{OBJ_TOKEN} {str(obj_idx)} : {KB_str} '
        out += obj_string

    return out, mask_idx

if __name__ == '__main__':
    dial_KB_to_string()