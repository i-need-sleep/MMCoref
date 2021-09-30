import os
import torch
from transformers import RobertaConfig, AutoModel
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from scripts.dataset_objIndex import make_loader_with_index
from scripts.focalloss import FocalLoss

def train():
    # Constant setup
    BATCH_SIZE = 16
    BATCH_SIZE_DEV = 4
    LR = 5e-7
    N_EPOCH = 15
    GAMMA = 0.3
    
    torch.manual_seed(21)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Make loaders
    train_loader = make_loader_with_index('train', BATCH_SIZE)
    dev_loader = make_loader_with_index('dev', BATCH_SIZE_DEV)

    # Setup Tensorboard
    writer = SummaryWriter(comment = f'RobertaOnBERT_Objidx batch_size={BATCH_SIZE}, Adam_lr={LR}, Gamma={GAMMA}')

    # Define the model
    # Add a linear layer to resize KB embeddigns to (batch_size, 512, 768).
    # Add the KB embeddings to the token embeddings
    # Add an embedding layer for object ids
    # Add a binary classification head
    class Roberta_on_KB_emb(torch.nn.Module):
        def __init__(self, device):
            super(Roberta_on_KB_emb, self).__init__()
            self.device = device
            self.config = RobertaConfig.from_pretrained('./pretrained/roberta-base')
            self.roberta = AutoModel.from_pretrained('./pretrained/roberta-base')
            # print(self.roberta)
            self.lin_KB = torch.nn.Linear(1024, 768)
            self.lin_clsHead = torch.nn.Linear(768, 1)
            self.obj_id_emb = torch.nn.Embedding(200, 768)
            self.emb_bag  = torch.nn.Linear(768*3, 768)

        def get_extended_attention_mask(self, attention_mask, input_shape, device): 
         """ 
         Makes broadcastable attention and causal masks so that future and masked tokens are ignored. 
  
         Arguments: 
             attention_mask (:obj:`torch.Tensor`): 
                 Mask with ones indicating tokens to attend to, zeros for tokens to ignore. 
             input_shape (:obj:`Tuple[int]`): 
                 The shape of the input to the model. 
             device: (:obj:`torch.device`): 
                 The device of the input to the model. 
  
         Returns: 
             :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`. 
         """ 
         # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length] 
         # ourselves in which case we just need to make it broadcastable to all heads. 
         if attention_mask.dim() == 3: 
             extended_attention_mask = attention_mask[:, None, :, :] 
         elif attention_mask.dim() == 2: 
             # Provided a padding mask of dimensions [batch_size, seq_length] 
             # - if the model is a decoder, apply a causal mask in addition to the padding mask 
             # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length] 
             if self.config.is_decoder: 
                 batch_size, seq_length = input_shape 
                 seq_ids = torch.arange(seq_length, device=device) 
                 causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None] 
                 # in case past_key_values are used we need to add a prefix ones mask to the causal mask 
                 # causal and attention masks must have same type with pytorch version < 1.3 
                 causal_mask = causal_mask.to(attention_mask.dtype) 
  
                 if causal_mask.shape[1] < attention_mask.shape[1]: 
                     prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1] 
                     causal_mask = torch.cat( 
                         [ 
                             torch.ones( 
                                 (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype 
                             ), 
                             causal_mask, 
                         ], 
                         axis=-1, 
                     ) 
  
                 extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :] 
             else: 
                 extended_attention_mask = attention_mask[:, None, None, :] 
         else: 
             raise ValueError( 
                 "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format( 
                     input_shape, attention_mask.shape 
                 ) 
             ) 
  
         # Since attention_mask is 1.0 for positions we want to attend and 0.0 for 
         # masked positions, this operation will create a tensor which is 0.0 for 
         # positions we want to attend and -10000.0 for masked positions. 
         # Since we are adding it to the raw scores before the softmax, this is 
         # effectively the same as removing these entirely. 
        #  extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility 
         extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0 
         return extended_attention_mask 
            
        def forward(self, tokens, attn_mask, KB_embs, obj_ids):
            KB_embs = self.lin_KB(KB_embs) # (batch, 512, 768)
            obj_id_embs = self.obj_id_emb(obj_ids)
            emb = self.roberta.embeddings(tokens)
            emb = self.emb_bag(torch.cat((emb, KB_embs, obj_id_embs), dim=2))

            extended_attn_mask = self.get_extended_attention_mask(attn_mask, emb.shape, self.device)
            h = self.roberta.encoder(emb, attention_mask=extended_attn_mask).last_hidden_state

            out = self.lin_clsHead(h).squeeze(dim=2)
            return out

    # Training setup
    model = Roberta_on_KB_emb(device).to(device)
    criterion = torch.nn.BCEWithLogitsLoss()#FocalLoss(gamma=GAMMA)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    # Eval for F1
    def eval(model):
        model.eval()
        with torch.no_grad():
            total_hit, total_pred_positive, total_truth_positive, total_loss, total_pred = 0, 0, 0, [], 0
            for idx, batch in enumerate(dev_loader):
                dial_tokens, roberta_attn_mask, obj_embs, obj_ids, output_mask, reference = (i.to(device) for i in batch)

                pred = model(dial_tokens, roberta_attn_mask, obj_embs, obj_ids)
                pred = pred[output_mask].reshape(-1)
                truth = reference.float()
                loss = criterion(pred, truth).detach()

                pred_bin = pred > 0
                truth_bin = truth > 0.5
                
                hit = torch.sum(pred_bin*truth_bin == 1).detach()
                pred_positive = torch.sum(pred > 0).detach()
                truth_positive = torch.sum(truth > 0.5).detach()

                total_loss.append(float(loss))
                total_hit += int(hit)
                total_pred_positive += int(pred_positive)
                total_truth_positive += int(truth_positive)
                total_pred += int(pred.shape[0])
            print('#pred positives',total_pred_positive)
            print('#groundtruth positives',total_truth_positive)
            print('#total pred', total_pred)
            print('#hit', total_hit)
            total_loss = sum(total_loss)/len(total_loss)
            if (total_pred_positive == 0):
                total_pred_positive = 1e10
            prec = total_hit / total_pred_positive
            recall = total_hit / total_truth_positive
            try:
                f1 = 2/(1/prec + 1/recall)
            except:
                f1 = 0
        return total_loss, prec, recall, f1

    # Train
    n_iter = 0
    n_prev_iter = 0
    running_loss = 0
    for epoch in range(N_EPOCH):
        for batch_idx, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            dial_tokens, roberta_attn_mask, obj_embs, obj_ids, output_mask, reference = (i.to(device) for i in batch)
            # truth = torch.ones_like(reference)
            truth = reference.float()

            with autocast():
                pred = model(dial_tokens, roberta_attn_mask, obj_embs, obj_ids)
                pred = pred[output_mask].reshape(-1)

                loss = criterion(pred, truth)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            n_iter += 1
            writer.add_scalar('Loss/train_batch', loss, n_iter)
            running_loss += loss.detach()

            if batch_idx % 500 == 0:
                print(pred)
                print(truth)
                print(running_loss/(n_iter-n_prev_iter))
                loss, prec, recall, f1 = eval(model)
                writer.add_scalar('Loss/train_avg', running_loss/(n_iter-n_prev_iter), n_iter)
                n_prev_iter = n_iter
                running_loss = 0
                writer.add_scalar('Loss/dev', loss, n_iter)
                writer.add_scalar('Precision/dev', prec, n_iter)
                writer.add_scalar('Recall/dev', recall, n_iter)
                writer.add_scalar('F1/dev', f1, n_iter)

                try:
                    os.makedirs('./checkpoint')
                except:
                    pass

                torch.save({
                    'epoch': epoch,
                    'step': n_iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dev_loss': loss,
                    }, f'./checkpoint/RobertaOnBERT_focal_mixed_ObjIdx_batchsize{BATCH_SIZE}_lr{LR}_Gamma{GAMMA}_{epoch}_{batch_idx}_{loss}_{f1}.bin')
    print('DONE !!!')

if __name__ == "__main__":
    train()