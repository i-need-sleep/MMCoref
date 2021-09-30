import torch
from transformers import RobertaConfig, RobertaModel
from torch.utils.tensorboard import SummaryWriter

from scripts.dataset import make_loader

# Constant setup
BATCH_SIZE = 16
DEV_BATCH_SIZE = 8
MAX_N_OBJ = 35

torch.manual_seed(21) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

LR = 1e-7
N_EPOCH = 3

# Get Loaders
train_loader = make_loader('balanced_train', BATCH_SIZE, MAX_N_OBJ)
dev_loader = make_loader('dev', DEV_BATCH_SIZE, MAX_N_OBJ)

# Load RoBERTa
configuration = RobertaConfig.from_pretrained('./roberta')
roberta = RobertaModel(configuration)
writer = SummaryWriter(comment = f'baselineRoberta batch_size={BATCH_SIZE}, Adam_lr={LR}')
print(configuration)

# Define the model (add a linear binary classification head)
class baseline_roberta(torch.nn.Module):
    def __init__(self, roberta):
        super(baseline_roberta, self).__init__()
        self.roberta = roberta
        self.class_head = torch.nn.Linear(768, 1)

    def forward(self, tokens, attn_mask):
        h = self.roberta(tokens, attn_mask)['last_hidden_state']
        out = self.class_head(h)
        return out

model = baseline_roberta(roberta).to(device)

# Training setup
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Eval for F1
def eval(model):
    model.eval()
    with torch.no_grad():
        total_hit, total_pred_positive, total_truth_positive, total_loss = 0, 0, 0, 0
        for idx, batch in enumerate(dev_loader):
            tokens, token_mask, target, target_mask, attn_mask = batch
            pred = model(tokens.to(device).long(), attn_mask.to(device).long())
            pred = pred.reshape(-1)[token_mask.to(device).reshape(-1)].float()
            truth = target.to(device).reshape(-1)[target_mask.to(device).reshape(-1)].float()
            loss = criterion(pred, truth)

            pred = torch.nn.Softmax()(pred.float())
            pred_bin = pred > 0.5
            truth_bin = truth > 0.5
            
            hit = torch.sum(pred_bin == truth_bin)
            pred_positive = torch.sum(pred > 0.5)
            truth_positive = torch.sum(truth > 0.5)

            total_loss += float(loss)
            total_hit += int(hit)
            total_pred_positive += int(pred_positive)
            total_truth_positive += int(truth_positive)

        print(total_pred_positive)
        prec = hit / total_pred_positive
        recall = hit / total_truth_positive
        f1 = 2/(1/prec + 1/recall)
    return loss, prec, recall, f1

# TRAIN !!!!
n_iter = 0
for epoch in range(N_EPOCH):
    for idx, batch in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        tokens, token_mask, target, target_mask, attn_mask = batch

        pred = model(tokens.to(device).long(), attn_mask.to(device).long())

        pred = pred.reshape(-1)[token_mask.to(device).reshape(-1)].float()
        truth = target.to(device).reshape(-1)[target_mask.to(device).reshape(-1)].float()
        print(pred)
        print(truth)
        print(torch.sum(truth))

        loss = criterion(pred, truth)
        loss.backward()
        optimizer.step()

        writer.add_scalar('Loss/train', loss)

        if idx % 500 == 0:
            n_iter += 500
            loss, prec, recall, f1 = eval(model)
            writer.add_scalar('Loss/train', loss, n_iter)
            writer.add_scalar('Precision/dev', prec, n_iter)
            writer.add_scalar('Recall/dev', recall, n_iter)
            writer.add_scalar('F1/dev', f1, n_iter)
            torch.save({
                'epoch': epoch,
                'step': idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dev_loss': loss,
                }, f'./checkpoint/baselineRoberta_balanced_batchsize{BATCH_SIZE}_lr{LR}_{epoch}_{idx}_{loss}_{f1}.bin')
print('DONE !!!')