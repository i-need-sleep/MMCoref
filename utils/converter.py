import json
import tqdm

DATA_ROOT = "../../data"
OUT_ROOT = "../processed"

def process_to_json(split = "dev", dial_mem = 3, out_path = 'example', test_data = False):

    # Output json:
    # Flattened dial hist
    # Obj: Formatted KB entry, one for each object

    dial_path = f"{DATA_ROOT}/simmc2_dials_dstc10_{split}.json"
    KB_fashion_path = f"{DATA_ROOT}/fashion_prefab_metadata_all.json"
    KB_furniture_path = f"{DATA_ROOT}/furniture_prefab_metadata_all.json"
    KB_dict_path = f"{OUT_ROOT}/KB_dict.json"
    
    out = []
    index_error = 0
    corrupted_error = 0
    step = 0
    
    with open(dial_path, 'r') as dial_file:
        dials = json.load(dial_file)['dialogue_data']

    with open(KB_fashion_path, 'r') as fashion_file:
        KB_fash = json.load(fashion_file)

    with open(KB_furniture_path, 'r') as furniture_file:
        KB_fur = json.load(furniture_file)

    with open(KB_dict_path, 'r') as KB_dict_file:
        KB_dict = json.load(KB_dict_file)

    for dial_idx, dial in enumerate(tqdm.tqdm(dials)):
        dial_hist = dial['dialogue']
        domain = dial['domain']
        scene_ids = dial['scene_ids']
        dial_idx = dial['dialogue_idx']
        
        for round_idx, cur_round in enumerate(dial_hist):

            round_out = '' # History for the round

            for r in range(max(0, round_idx - dial_mem), round_idx):
                #Loop over the dial history
                round = dial_hist[r]
                round_out += f'User : {round["transcript"]} System : {round["system_transcript"]} '
                system_mentions = round["system_transcript_annotated"]['act_attributes']['objects']
                if len(system_mentions) != 0:
                    round_out += f'System mentions : {" and ".join([str(mention) for mention in system_mentions])} '

            round_out += f'User : {cur_round["transcript"]}'
        
            #  Fetch KB entries for all candidate objects

            # Skip if the reference is ambiguious
            try:
                if cur_round['disambiguation_label']:
                    continue
            except:
                pass

            if domain == 'fashion':
                KB = KB_fash
            else:
                KB = KB_fur

            scene_paths = []
            for scene_path in scene_ids.values():
                scene_paths.append(scene_path)

            if not test_data:            
                references = cur_round['transcript_annotated']['act_attributes']['objects']
            else:
                references = []

            # Resolve current scene path
            scene_key = 0
            for (key, path) in scene_ids.items():
                if int(key) >= scene_key and int(key) <= round_idx:
                    scene_key = int(key)

            objects, reference_mask, reference_index, candidate_ids, candidate_pos, candidate_bbox, KB_ids, error, candidate_relations, scene_seg = get_object_KBs(scene_paths, KB, domain, references, KB_dict, scene_key)
            
            if error:
                print(dial_idx, round_idx, step)
                index_error += 1
            
            step += 1

            # Handle corrupted scene images
            if not test_data and ('cloth_store_1416238_woman_4_8' in scene_paths or 'm_cloth_store_1416238_woman_20_6' in scene_paths or 'cloth_store_1416238_woman_20_6' in scene_paths or 'cloth_store_1416238_woman_19_0' in scene_paths):
                corrupted_error += 1
                continue

            out.append({
                'dial': round_out, 
                'objects': objects, 
                'reference_mask': reference_mask, 
                'reference_idx': reference_index, 
                'candidate_ids': candidate_ids,
                'candidate_pos': candidate_pos,
                'candidate_bbox': candidate_bbox,
                'candidate_relations': candidate_relations,
                'scene_seg': scene_seg,
                'KB_ids': KB_ids,
                'scenes': scene_paths,
                'domain': domain,
                'dial_idx': dial_idx,
                'round_idx': round_idx
            })        
       
        # if dial_idx > 10:
        #     break

    with open(f'{OUT_ROOT}/{out_path}.json', 'w', encoding='utf-8') as out_file:
        json.dump(out, out_file)
    print(f'# index error: {index_error}, # corrupted error {corrupted_error}, out length: {len(out)}')
    return

def get_object_KBs(scene_paths, KB, domain, references, KB_dict, cur_scene_idx):
    scene_root = f"{DATA_ROOT}/simmc2_scene_jsons_dstc10_public/public"
    scene_data_lst = []
    scene_rel_data = []
    used_idx = []

    for scene_path in scene_paths:
        with open(f'{scene_root}/{scene_path}_scene.json', 'r') as f:
            data = json.load(f)
            scene_data_lst.append(data['scenes'][0]['objects'])
            scene_rel_data.append(data['scenes'][0]['relationships'])

    candidate_relations = {'left':{}, 'right': {}, 'up': {}, 'down':{}}
    for scene_rel in scene_rel_data:
        for rel in ['left', 'right', 'up', 'down']:
            if rel in scene_rel.keys():
                for rel_key, rel_val in scene_rel[rel].items():
                    candidate_relations[rel][rel_key] = rel_val

    out = []
    reference_mask = []
    reference_index = []
    candidate_ids = []
    candidate_pos = []
    candidate_bbox = []
    KB_ids = []
    scene_seg = []
    error = False
    for scene_idx, scene_data in enumerate(scene_data_lst):
        for i, object in enumerate(scene_data):
            object_idx = object['index']
            bbox = object['bbox']
            position = object['position']

            object_KB = KB[object['prefab_path']]

            if object_idx in used_idx:
                continue # in the case of multiple scenes containing the same object
            else:
                used_idx.append(object_idx)

            if domain == 'fashion':

                object_string = f'''
                        Item {object_idx} is located at x : {'{:.2f}'.format(float(position[0]))}, y : {'{:.2f}'.format(float(position[1]))}, z: {'{:.2f}'.format(float(position[2]))}.
                        Its located in the bounding box {' '.join([str(x) for x in bbox])}.
                        Its price is {object_KB['price']}.
                        Its size is {object_KB['size']}.
                        Its brand is {object_KB['brand']}.
                        It has a customer review of {object_KB['customerReview']} out of 5.
                        It is available in sizes {' and '.join(object_KB['availableSizes'])}.
                    '''
            else:
                object_string = f'''
                        Item {object_idx} is located at x : {'{:.2f}'.format(float(position[0]))}, y : {'{:.2f}'.format(float(position[1]))}, z: {'{:.2f}'.format(float(position[2]))}.
                        Its located in the bounding box {' '.join([str(x) for x in bbox])}.
                        Its price is {object_KB['price']}.
                        Its brand is {object_KB['brand']}.
                        It is made with {object_KB['materials']}.
                        It has a customer review of {object_KB['customerRating']} out of 5.
                    '''
            object_string = object_string.split('\n')
            object_string = ' '.join([line.strip() for line in object_string]).strip()

            out.append(object_string)
            candidate_ids.append(object_idx)
            candidate_pos.append(position)
            candidate_bbox.append(bbox)
            KB_ids.append(KB_dict[object['prefab_path']])

            if object_idx in references:
                reference_mask.append(1)
                reference_index.append(i)
            else:
                reference_mask.append(0)
            
            if (scene_idx == 0 and cur_scene_idx == 0) or (scene_idx != 0 and cur_scene_idx != 0):
                scene_seg.append(1)
            else:
                scene_seg.append(2)

    # Account for scene feats
    if cur_scene_idx == 0:
        scene_seg += [1,2]
    else:
        scene_seg += [2,1]

    if len(reference_index) != len(references):
        # print(reference_index, references)
        # error = True
        pass # duplicate references
    return out, reference_mask, reference_index, candidate_ids, candidate_pos, candidate_bbox, KB_ids, error, candidate_relations, scene_seg
        

if __name__ == '__main__':
    process_to_json(split='dev', out_path='dev')
    process_to_json(split='train', out_path='train')
    process_to_json(split='devtest', out_path='devtest')
    process_to_json(split='teststd_public', out_path='teststd', test_data=True)