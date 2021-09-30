import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer

PROCESSED_ROOT = './processed'

class BaselineDataset(Dataset):
    def __init__(self, split, max_len):
        self.file_path = f'{PROCESSED_ROOT}/{split}.txt'
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.max_len = max_len

        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        [text, target_ids] = self.data[index].split(' , => Predict :')
        target_ids = target_ids.split('\n')[0]
        tokenized = self.tokenizer(text, padding='max_length') 

        tokens = np.array(tokenized['input_ids'])
        attn_masks = np.array(tokenized['attention_mask'])
        token_mask = tokens == self.tokenizer.mask_token_id
        target = np.zeros((self.max_len))
        if target_ids.strip() != '':
            target_ids = target_ids.strip().split(' ')
            target_ids = np.array([int(i) for i in target_ids])
            target[target_ids] += 1
        target_mask = np.zeros((self.max_len))
        target_mask[:tokens[token_mask].shape[0]] = 1
        target_mask = target_mask == 1
        return tokens, token_mask, target, target_mask, attn_masks


def make_loader(split, batch_size, max_len):
    '''
    tokens, token_mask, target, target_mask for each line
    token[token_mask] should be aligned with target[target_mask]
    '''
    dataset = BaselineDataset(split, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

if __name__ == "__main__":
    loader = make_loader('balanced_train', 3, 35)
    for idx, data in enumerate(loader):
        pass
    print('done')
        