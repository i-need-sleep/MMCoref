import torch
import math
from .vqa_model import VQAModel

class Clean_Uniter(torch.nn.Module):
    def __init__(self):
        super(Clean_Uniter, self).__init__()
        # Load pretrained UNITER
        self.uniter = VQAModel(num_answers = 69, model = 'uniter')
        self.uniter.encoder.load('./Transformers_VQA/models/pretrained/uniter-base.pt')

        # Convert input (Ojb_idx: 512, KB: 1024, visual: 512, pos: 3*128)
        self.lin_vis = torch.nn.Linear(512, 2048)

        # Convert output
        self.clsHead = torch.nn.Linear(768, 1)
            
    def forward(self, input_ids , txt_seg_ids, vis_feats, obj_embs, obj_ids, pos_x, pos_y, pos_z, bboxes, vis_seg, extended_attention_mask):
        # combine object features
        
        vis_feats = self.lin_vis(vis_feats.float())

        word_embeddings = self.uniter.encoder.model.uniter.embeddings(input_ids, txt_seg_ids)
        img_type_embeddings = self.uniter.encoder.model.uniter.embeddings.token_type_embeddings(vis_seg)
        img_embeddings = self.uniter.encoder.model.uniter.img_embeddings(vis_feats, bboxes, img_type_embeddings)
        embeddings = torch.cat([word_embeddings,img_embeddings],dim=1)

        lang_v_feats = self.uniter.encoder.model.uniter.encoder(hidden_states = embeddings, attention_mask = extended_attention_mask)
        
        out = self.clsHead(lang_v_feats)
        return out