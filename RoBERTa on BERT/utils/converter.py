import json
from re import split
import tqdm

DATA_ROOT = "../../data"
OUT_ROOT = "../processed"

def process_to_json(split = "dev", dial_mem = 3, out_path = 'example'):

    # Output json:
    # Flattened dial hist
    # Obj: Formatted KB entry, one for each object

    dial_path = f"{DATA_ROOT}/simmc2_dials_dstc10_{split}.json"
    KB_fashion_path = f"{DATA_ROOT}/fashion_prefab_metadata_all.json"
    KB_furniture_path = f"{DATA_ROOT}/furniture_prefab_metadata_all.json"
    
    out = []
    index_error = 0
    
    with open(dial_path, 'r') as dial_file:
        dials = json.load(dial_file)['dialogue_data']

    with open(KB_fashion_path, 'r') as fashion_file:
        KB_fash = json.load(fashion_file)

    with open(KB_furniture_path, 'r') as furniture_file:
        KB_fur = json.load(furniture_file)

    for dial_idx, dial in enumerate(tqdm.tqdm(dials)):
        dial_hist = dial['dialogue']
        domain = dial['domain']
        scene_ids = dial['scene_ids']
        
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

            scene_path = ''
            scene_key = 0

            # Resolve scene path
            for (key, path) in scene_ids.items():
                if int(key) >= scene_key and int(key) <= round_idx:
                    scene_key = int(key)
                    scene_path = path

            references = cur_round['transcript_annotated']['act_attributes']['objects']
            objects, reference_mask, reference_index, error = get_object_KBs(scene_path, KB, domain, references)
            
            if error:
                index_error += 1
                continue


            out.append({'dial': round_out, 'objects': objects, 'reference_mask': reference_mask, 'reference_idx': reference_index})        
        # if dial_idx > 5:
        #     break

    with open(f'{OUT_ROOT}/{out_path}.json', 'w', encoding='utf-8') as out_file:
        json.dump(out, out_file)
    print(f'# index error: {index_error}, no error: {len(out)}')
    return

def get_object_KBs(scene_path, KB, domain, references):
    scene_root = f"{DATA_ROOT}/simmc2_scene_jsons_dstc10_public/public"

    with open(f'{scene_root}/{scene_path}_scene.json', 'r') as f:
        scene_data = json.load(f)['scenes'][0]['objects']

    out = []
    reference_mask = []
    reference_index = []
    error = False
    for i, object in enumerate(scene_data):
        object_idx = object['index']
        bbox = object['bbox']
        position = object['position']

        object_KB = KB[object['prefab_path']]

        if domain == 'fashion':
            if object_KB['sleeveLength'] != '':
                sleeveLen_str = f"It has {object_KB['sleeveLength']} sleeve length."
            else:
                sleeveLen_str = ''

            object_string = f'''
                    Item {object_idx} is a {object_KB['type']}. 
                    It is located at x : {'{:.2f}'.format(float(position[0]))}, y : {'{:.2f}'.format(float(position[1]))}, z: {'{:.2f}'.format(float(position[2]))}.
                    Its located in the bounding box {' '.join([str(x) for x in bbox])}.
                    Its price is {object_KB['price']}.
                    Its size is {object_KB['size']}.
                    {sleeveLen_str}
                    Its brand is {object_KB['brand']}.
                    Its has {object_KB['pattern']} pattern.
                    Its is {' and '.join(object_KB['color'].split(', '))} in color.
                    It has a customer review of {object_KB['customerReview']} out of 5.
                    It is available in sizes {' and '.join(object_KB['availableSizes'])}.
                '''
        else:
            object_string = f'''
                    Item {object_idx} is a {object_KB['type']}. 
                    It is located at x : {'{:.2f}'.format(float(position[0]))}, y : {'{:.2f}'.format(float(position[1]))}, z: {'{:.2f}'.format(float(position[2]))}.
                    Its located in the bounding box {' '.join([str(x) for x in bbox])}.
                    Its price is {object_KB['price']}.
                    Its brand is {object_KB['brand']}.
                    It is made with {object_KB['materials']}.
                    Its is {object_KB['color']} in color.
                    It has a customer review of {object_KB['customerRating']} out of 5.
                '''
        object_string = object_string.split('\n')
        object_string = ' '.join([line.strip() for line in object_string]).strip()
        out.append(object_string)

        if object_idx in references:
            reference_mask.append(1)
            reference_index.append(i)
        else:
            reference_mask.append(0)

    if len(reference_index) != len(references):
        error = True
    return out, reference_mask, reference_index, error
        

if __name__ == '__main__':
    process_to_json(split='devtest', out_path='devtest')