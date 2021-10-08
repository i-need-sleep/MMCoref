import torch
import math
from .vqa_model import VQAModel

class ObjPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model = 128, dropout = 0.1, max_len = 10002):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
            x: (#batch, #seq_length)
        """
        pos = x * 5
        pos += 5001
        pos = torch.round(pos.float()).long()
        out = self.pe[pos]

        # padding pos returns a 0 vertor
        out[[x==0]] = 0
        return self.dropout(out) # (batch, len_seq, 128)

class Modified_Uniter_attnbias_rcnn_SBERT(torch.nn.Module):
    def __init__(self):
        super(Modified_Uniter_attnbias_rcnn_SBERT, self).__init__()
        # Load pretrained UNITER
        self.uniter = VQAModel(num_answers = 69, model = 'uniter')
        self.uniter.encoder.load('./Transformers_VQA/models/pretrained/uniter-base.pt')

        # Convert input (Ojb_idx: 512, KB: 768, visual: 512+2048, pos: 3*128, scene_seg: 128)
        self.obj_idx_emb = torch.nn.Embedding(200, 512, padding_idx=0)
        self.scene_seg_emb = torch.nn.Embedding(3, 128, padding_idx=0)
        self.obj_pos_enc = ObjPositionalEncoding()
        self.emb_bag = torch.nn.Linear(2048+256+2048, 2048)

        # Convert output
        self.clsHead = torch.nn.Linear(768, 1)
            
    def forward(self, input_ids , txt_seg_ids, vis_feats, obj_embs, obj_ids, pos_x, pos_y, pos_z, bboxes, vis_seg, extended_attention_mask, scene_seg, rel_masks):
        # combine object features
        obj_id_embs = self.obj_idx_emb(obj_ids)
        pos_x_emb = self.obj_pos_enc(pos_x)
        pos_y_emb = self.obj_pos_enc(pos_y)
        pos_z_emb = self.obj_pos_enc(pos_z)
        scene_seg_emb = self.scene_seg_emb(scene_seg)
        vis_feats = torch.cat((vis_feats, obj_embs, obj_id_embs, pos_x_emb, pos_y_emb, pos_z_emb, scene_seg_emb), axis=2)
        vis_feats = self.emb_bag(vis_feats)

        word_embeddings = self.uniter.encoder.model.uniter.embeddings(input_ids, txt_seg_ids)
        img_type_embeddings = self.uniter.encoder.model.uniter.embeddings.token_type_embeddings(vis_seg)
        img_embeddings = self.uniter.encoder.model.uniter.img_embeddings(vis_feats, bboxes, img_type_embeddings)
        embeddings = torch.cat([word_embeddings,img_embeddings],dim=1)

        lang_v_feats = self.uniter.encoder.model.uniter.encoder(hidden_states = embeddings, attention_mask = extended_attention_mask, rel_masks=rel_masks)
        
        out = self.clsHead(lang_v_feats)
        return out

if __name__ == '__main__':
    pass