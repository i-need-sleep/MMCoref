import json
from re import L
import torch
from torch.utils.data import Dataset, DataLoader
from .src.tokenization import BertTokenizer

PROCESSED_ROOT = './processed'

class UNITER_on_CLIP_BERT_Dataset(Dataset):
    def __init__(self, split, max_n_obj=200):
        self.file_path = f'{PROCESSED_ROOT}/{split}.json'
        self.KB_emb_path = f'{PROCESSED_ROOT}/KB_{split}.pt'
        self.vis_feat_path = f'{PROCESSED_ROOT}/img_features.pt'
        self.KB_dict_path = f'{PROCESSED_ROOT}/KB_dict.json'
        self.max_n_obj = max_n_obj
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = BertTokenizer.from_pretrained('./pretrained/bert-base-cased/bert-base-cased-vocab.txt')

        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        with open(self.KB_dict_path, 'r', encoding='utf-8') as f:
            self.KB_dict = json.load(f)

        self.KB_emb_dict = torch.load(self.KB_emb_path, map_location=self.device) # KB_emb_dict['obj_string']
        self.vis_feat_dict = torch.load(self.vis_feat_path, map_location=self.device) # vis_feat_dict[scene_name][local_idx or 'Scene']
    
    def __len__(self):
        return len(self.data)
    
    def _uniterBoxes(self, boxes):#uniter requires a 7-dimensiom beside the regular 4-d bbox. From the Transformer VQA repo
        new_boxes = torch.zeros(boxes.shape[0],7)
        new_boxes = torch.zeros(boxes.shape[0],7)
        new_boxes[:,1] = boxes[:,0]
        new_boxes[:,0] = boxes[:,1]
        new_boxes[:,3] = boxes[:,2]
        new_boxes[:,2] = boxes[:,3]
        new_boxes[:,4] = new_boxes[:,3]-new_boxes[:,1] #w
        new_boxes[:,5] = new_boxes[:,2]-new_boxes[:,0] #h
        new_boxes[:,6]=new_boxes[:,4]*new_boxes[:,5] #area
        return new_boxes  

    def _make_relationship_mask(self, obj_rels, obj_ids):
        out = []
        for rel in ['left', 'right', 'up', 'down']:
            rel_dict = obj_rels[rel]
            mask = torch.zeros(self.max_n_obj, self.max_n_obj)
            for idx, obj_id in enumerate(obj_ids):
                if str(obj_id) in rel_dict.keys():
                    for val in rel_dict[str(obj_id)]:
                        mask[idx][obj_ids.index(val)] = 1
            mask = torch.nn.ZeroPad2d((512 - self.max_n_obj, 0, 0, 512 - self.max_n_obj))(mask) # zero pad to (512, 512)
            mask = mask.unsqueeze(0)
            out.append(mask)
        return out

    def __getitem__(self, index):
        line = self.data[index]
        dial, objects, reference, obj_ids, obj_pos, obj_bbox, scenes, KB_ids, scene_segs, obj_rels = line['dial'], line['objects'], line['reference_mask'], line['candidate_ids'], line['candidate_pos'], line['candidate_bbox'], line['scenes'], line['KB_ids'], line['scene_seg'], line['candidate_relations']
        
        # relationship mask (512, 512) * 4
        rel_mask_left, rel_mask_right, rel_mask_up, rel_mask_down = self._make_relationship_mask(obj_rels, obj_ids)

        # obj_embs
        object_string = json.dumps(objects)
        obj_embs = self.KB_emb_dict[object_string] # (#object, 1024)
        obj_embs = torch.nn.ZeroPad2d((0,0,0, self.max_n_obj - obj_embs.shape[0]))(obj_embs) # zero pad to (max_n_obj, 1024)

        # scene_segs (1, #obj + 2)
        scene_segs = torch.tensor([scene_segs])
        scene_segs = torch.nn.ZeroPad2d((0, self.max_n_obj - scene_segs.shape[1],0,0))(scene_segs) # zero pad to (1, max_n_obj)

        # KB_ids (1, #obj)
        KB_ids = torch.tensor([KB_ids])
        KB_ids = torch.nn.ZeroPad2d((0, self.max_n_obj - KB_ids.shape[1],0,0))(KB_ids) # zero pad to (1, max_n_obj)

        # Merge vis feats of all scenes
        vis_feat_scenes = {}
        scene_feats = []
        for scene in scenes:
            vis_feat_scene = self.vis_feat_dict[scene+'_scene']
            for key, val in vis_feat_scene.items():
                vis_feat_scenes[key] = val
            scene_feats.append(vis_feat_scene['scene'])
        
        # Make the vis feat tensor (#obj + #scene, 512)
        for _, obj_id in enumerate(obj_ids):
            if _ == 0:
                vis_feats = vis_feat_scenes[obj_id]
            else:
                vis_feats = torch.cat((vis_feats, vis_feat_scenes[obj_id]), axis=0)
        for scene_feat in scene_feats:
            vis_feats = torch.cat((vis_feats, scene_feat), axis=0) 
        vis_seg = torch.ones_like(vis_feats[:,0]).unsqueeze(0)
        vis_mask = torch.ones_like(vis_feats[:,0]).unsqueeze(0)
        vis_feats = torch.nn.ZeroPad2d((0,0,0, self.max_n_obj - vis_feats.shape[0]))(vis_feats) # zero pad to (max_n_obj, 512)
        vis_seg = torch.nn.ZeroPad2d((0, self.max_n_obj - vis_seg.shape[1],0,0))(vis_seg).long() # zero pad to (1, max_n_obj)
        vis_mask = torch.nn.ZeroPad2d((0,self.max_n_obj - vis_mask.shape[1],0,0))(vis_mask)
        
        # Obj pos (1, #obj)
        pos_x = []
        pos_y = []
        pos_z = []
        for pos in obj_pos:
            pos_x.append(pos[0])
            pos_y.append(pos[1])
            pos_z.append(pos[2])
        pos_x = torch.nn.ZeroPad2d((0, self.max_n_obj - len(obj_ids), 0, 0))(torch.tensor(pos_x).reshape(1,-1)) # zero pad to (1, max_n_obj)
        pos_y = torch.nn.ZeroPad2d((0, self.max_n_obj - len(obj_ids), 0, 0))(torch.tensor(pos_y).reshape(1,-1))
        pos_z = torch.nn.ZeroPad2d((0, self.max_n_obj - len(obj_ids), 0, 0))(torch.tensor(pos_z).reshape(1,-1))

        # Bboxes (#obj, 7)
        for _, boxes in enumerate(obj_bbox):
            if _ == 0:
                bboxes = torch.tensor([boxes])
            else:
                bboxes = torch.cat((bboxes, torch.tensor([boxes])), axis=0)
        bboxes = torch.nn.ZeroPad2d((0,0,0,self.max_n_obj-bboxes.shape[0]))(bboxes) # zero pad to (max_n_obj, 4) x,y,h,w
        new_bboxes = torch.zeros_like(bboxes)
        new_bboxes[:,0:2] = bboxes[:,0:2] 
        new_bboxes[:,2] = bboxes[:,0] + bboxes[:,3]
        new_bboxes[:,3] = bboxes[:,1] + bboxes[:,2] # x1, y1, x2, y2
        bboxes = self._uniterBoxes(new_bboxes) # convert to uniter's bbox representation (max_n_obj, 7)

        # Obj_ids (1, #obj)
        obj_ids = torch.tensor([obj_ids])
        obj_ids += 1 # 0 is reserved for padding
        output_mask = torch.ones_like(obj_ids)
        obj_ids = torch.nn.ZeroPad2d((0, self.max_n_obj - obj_ids.shape[1], 0, 0))(obj_ids) # zero pad to (1, max_n_obj)
        output_mask = torch.nn.ZeroPad2d((512 - self.max_n_obj, self.max_n_obj - output_mask.shape[1], 0, 0))(output_mask) # zero pad to (1, 512)

        # Input ids (512-max_n_obj)
        max_n_ids = 512 - self.max_n_obj
        tokens_a = self.tokenizer.tokenize(dial.strip())
        # Brutally handle the over-sized text inputs
        if len(tokens_a) > 510-self.max_n_obj:
            # print(len(tokens_a))
            tokens_a = tokens_a[510-self.max_n_obj-len(tokens_a):]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        if len(tokens) > 512-self.max_n_obj:
            print(len(tokens))
            print('TOO LOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOONG')
            raise 
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        segment_ids = [0] * len(tokens)
        input_mask = [1] * len(input_ids)

        # Zero pad
        padding = [0] * (512-self.max_n_obj - len(input_ids))
        
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        input_ids = torch.tensor([input_ids])
        input_mask = torch.tensor([input_mask])
        txt_seg_ids = torch.tensor([segment_ids])

        attn_mask = torch.cat((input_mask.to(self.device), vis_mask.to(self.device)), axis=1)
        extended_attention_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        reference = torch.tensor(reference)

        round_idx = line['round_idx']
        dial_idx = line['dial_idx']

        return input_ids , txt_seg_ids, vis_feats, KB_ids, obj_ids, pos_x, pos_y, pos_z, bboxes, vis_seg, extended_attention_mask, output_mask, reference, dial_idx, round_idx, obj_embs, scene_segs, rel_mask_left, rel_mask_right, rel_mask_up, rel_mask_down

def mr_collate(data):
    
    input_ids , txt_seg_ids, vis_feats, KB_ids, obj_ids, pos_x, pos_y, pos_z, bboxes, vis_seg, extended_attention_mask, output_mask, reference, dial_idx, round_idx, obj_embs, scene_segs, rel_mask_left, rel_mask_right, rel_mask_up, rel_mask_down = data[0]
    vis_feats = vis_feats.unsqueeze(0)
    bboxes = bboxes.unsqueeze(0)
    obj_embs = obj_embs.unsqueeze(0)
    dial_idx = [dial_idx]
    round_idx = [round_idx]

    for idx, line in enumerate(data):
        if idx == 0:
            continue
        input_ids_l , txt_seg_ids_l, vis_feats_l, KB_ids_l, obj_ids_l, pos_x_l, pos_y_l, pos_z_l, bboxes_l, vis_seg_l, extended_attention_mask_l, output_mask_l, reference_l, dial_idx_l, round_idx_l, obj_embs_l, scene_segs_l, rel_mask_left_l, rel_mask_right_l, rel_mask_up_l, rel_mask_down_l = line 
        vis_feats_l = vis_feats_l.unsqueeze(0)
        bboxes_l = bboxes_l.unsqueeze(0)
        obj_embs_l = obj_embs_l.unsqueeze(0)
        
        input_ids = torch.cat((input_ids, input_ids_l), dim=0)
        txt_seg_ids = torch.cat((txt_seg_ids, txt_seg_ids_l), dim=0)
        vis_feats  = torch.cat((vis_feats, vis_feats_l), dim=0)
        KB_ids = torch.cat((KB_ids, KB_ids_l), axis=0)
        obj_ids = torch.cat((obj_ids, obj_ids_l), dim=0)
        pos_x = torch.cat((pos_x, pos_x_l), dim=0)
        pos_y = torch.cat((pos_y, pos_y_l), dim=0)
        pos_z = torch.cat((pos_z, pos_z_l), dim=0)
        bboxes = torch.cat((bboxes, bboxes_l), dim=0)
        vis_seg = torch.cat((vis_seg, vis_seg_l), dim=0)
        extended_attention_mask = torch.cat((extended_attention_mask, extended_attention_mask_l), dim=0)
        output_mask = torch.cat((output_mask, output_mask_l), dim=1)
        reference = torch.cat((reference,reference_l), dim=0)
        dial_idx.append(dial_idx_l)
        round_idx.append(round_idx_l)
        obj_embs = torch.cat((obj_embs, obj_embs_l), dim=0)
        scene_segs = torch.cat((scene_segs, scene_segs_l), dim=0)
        rel_mask_left = torch.cat((rel_mask_left, rel_mask_left_l), dim=0)
        rel_mask_right = torch.cat((rel_mask_right, rel_mask_right_l), dim=0)
        rel_mask_up = torch.cat((rel_mask_up, rel_mask_up_l), dim=0)
        rel_mask_down = torch.cat((rel_mask_down, rel_mask_down_l), dim=0)
        
    return {
        'input_ids': input_ids,
        'txt_seg_ids': txt_seg_ids, 
        'vis_feats': vis_feats, 
        'KB_ids': KB_ids, 
        'obj_ids': obj_ids, 
        'pos_x': pos_x, 
        'pos_y': pos_y, 
        'pos_z': pos_z, 
        'bboxes': bboxes, 
        'vis_seg': vis_seg, 
        'extended_attention_mask': extended_attention_mask, 
        'output_mask': output_mask, 
        'reference': reference, 
        'dial_idx': dial_idx,        
        'round_idx': round_idx,
        'obj_embs': obj_embs,
        'scene_segs': scene_segs,
        'rel_mask_left': rel_mask_left,
        'rel_mask_right': rel_mask_right,
        'rel_mask_up': rel_mask_up,
        'rel_mask_down': rel_mask_down
    }
    
def make_final_loader(split, batch_size):
    dataset = UNITER_on_CLIP_BERT_Dataset(split)
    loader = DataLoader(dataset, batch_size=batch_size ,shuffle=False, collate_fn=mr_collate)
    return loader

if __name__ == "__main__":
    pass
        