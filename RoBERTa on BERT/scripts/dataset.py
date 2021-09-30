import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, BertTokenizer

PROCESSED_ROOT = './processed'

class RoBERTa_on_BERT_Dataset(Dataset):
    def __init__(self, split):
        self.file_path = f'{PROCESSED_ROOT}/{split}.json'
        self.emb_path = f'{PROCESSED_ROOT}/KB_{split}.pt'
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained("./pretrained/roberta-base")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.emb_dict = torch.load(self.emb_path, map_location=self.device)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data[index]
        dial, objects, reference = line['dial'], line['objects'], line['reference_mask']

        dial_token_len = len(self.roberta_tokenizer(dial)['input_ids'])

        dial += self.roberta_tokenizer.eos_token
        for i in range(len(objects)):
            dial += self.roberta_tokenizer.cls_token
        dial_tokenized = self.roberta_tokenizer(dial, padding='max_length')
        dial_tokens, roberta_attn_mask = dial_tokenized['input_ids'], dial_tokenized['attention_mask']

        output_mask = torch.zeros_like(torch.tensor(dial_tokens))
        output_mask[dial_token_len: dial_token_len + len(objects)] = 1
        output_mask = output_mask == 1

        object_string = json.dumps(objects)
        obj_embs = self.emb_dict[object_string]
        obj_embs = torch.nn.ZeroPad2d((0,0,dial_token_len, len(dial_tokens) - dial_token_len - obj_embs.shape[0]))(obj_embs)

        # for i in (dial_tokens, roberta_attn_mask, obj_embs, output_mask, reference):
        #     print(torch.tensor(i).shape)    

        return dial_tokens, roberta_attn_mask, obj_embs, output_mask, reference


def make_loader(split, batch_size):
    dataset = RoBERTa_on_BERT_Dataset(split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=mr_collate)
    return loader

def mr_collate(data):
    dial_tokens, roberta_attn_mask, obj_embs, output_mask, reference = (torch.tensor(i) for i in data[0])
    dial_tokens = dial_tokens.unsqueeze(0)
    roberta_attn_mask = roberta_attn_mask.unsqueeze(0)
    obj_embs = obj_embs.unsqueeze(0)
    output_mask = output_mask.unsqueeze(0)
    
    for idx, line in enumerate(data):
        if idx == 0:
            continue
        dial_tokens_l, roberta_attn_mask_l, obj_embs_l, output_mask_l, reference_l = (torch.tensor(i) for i in line)
        dial_tokens_l = dial_tokens_l.unsqueeze(0)
        roberta_attn_mask_l = roberta_attn_mask_l.unsqueeze(0)
        obj_embs_l = obj_embs_l.unsqueeze(0)
        output_mask_l = output_mask_l.unsqueeze(0)

        dial_tokens = torch.cat((dial_tokens, dial_tokens_l), dim=0)
        roberta_attn_mask = torch.cat((roberta_attn_mask, roberta_attn_mask_l), dim=0)
        obj_embs = torch.cat((obj_embs, obj_embs_l), dim=0)
        output_mask = torch.cat((output_mask, output_mask_l), dim=0)
        reference = torch.cat((reference, reference_l), dim=0)

    # for i in (dial_tokens, roberta_attn_mask, obj_embs, output_mask, reference):
    #     print(i.shape)
    # print(torch.sum(output_mask))    
    return dial_tokens, roberta_attn_mask, obj_embs, output_mask, reference

if __name__ == "__main__":
    loader = make_loader('dev', 16)
    for i, s in enumerate(loader):
        pass
    print('DONE')
        