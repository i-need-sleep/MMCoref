import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from scripts.focalloss import FocalLoss

from Transformers_VQA.dataset_final import make_final_loader
from Transformers_VQA.modified_uniter_attnbias_rcnn import Modified_Uniter_attnbias_rcnn

def train():
    # Constant setup
    BATCH_SIZE = 3
    BATCH_SIZE_DEV = 1
    LR = 5e-6
    N_EPOCH = 30
    GAMMA = 2
    ALPHA = 5
    print(f'UNITERonCLIPBERT_attnbias_rcnn batch_size={BATCH_SIZE}, Adam_lr={LR}, FocalAlpha={ALPHA}, GAMMA={GAMMA}')

    torch.manual_seed(21)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Make loaders
    train_loader = make_final_loader('train', BATCH_SIZE, rcnn=True)
    dev_loader = make_final_loader('dev', BATCH_SIZE_DEV, rcnn=True)

    # Setup Tensorboard
    writer = SummaryWriter(comment = f'UNITERonCLIPBERT_attnbias_rcnn batch_size={BATCH_SIZE}, Adam_lr={LR}, FocalAlpha={ALPHA}, GAMMA={GAMMA}')

    # multiply by layer, do an embedding for indices 1 to 12
    mask_stepper = torch.ones(1, 12, 512, 512).to(device)
    for i in range(12):
        mask_stepper[0, i, :, :] *= i+1

    # Eval for F1
    def eval(model):
        model.eval()
        with torch.no_grad():
            total_hit, total_pred_positive, total_truth_positive, total_loss, total_pred = 0, 0, 0, [], 0
            for idx, batch in enumerate(dev_loader):
                input_ids = batch['input_ids'].to(device)
                txt_seg_ids = batch['txt_seg_ids'].to(device)
                vis_feats = batch['vis_feats'].to(device)
                obj_embs = batch['obj_embs'].to(device)
                obj_ids = batch['obj_ids'].to(device)
                pos_x = batch['pos_x'].to(device)
                pos_y = batch['pos_y'].to(device)
                pos_z = batch['pos_z'].to(device)
                bboxes = batch['bboxes'].to(device)
                vis_seg = batch['vis_seg'].to(device)
                extended_attention_mask = batch['extended_attention_mask'].to(device)
                output_mask = batch['output_mask'].to(device)
                reference = batch['reference'].to(device)
                scene_seg = batch['scene_segs'].to(device)
                rel_mask_left = batch['rel_mask_left'].to(device).unsqueeze(0).unsqueeze(2)
                rel_mask_right = batch['rel_mask_right'].to(device).unsqueeze(0).unsqueeze(2)
                rel_mask_up = batch['rel_mask_up'].to(device).unsqueeze(0).unsqueeze(2)
                rel_mask_down = batch['rel_mask_down'].to(device).unsqueeze(0).unsqueeze(2)

                rel_masks = torch.cat((rel_mask_left, rel_mask_right, rel_mask_up, rel_mask_down), axis=0)
                rel_masks = rel_masks * mask_stepper

                pred = model(input_ids , txt_seg_ids, vis_feats, obj_embs, obj_ids, pos_x, pos_y, pos_z, bboxes, vis_seg, extended_attention_mask, scene_seg, rel_masks)
                pred = pred.reshape(1,-1)
                pred = pred[output_mask==1].reshape(-1,1)
                truth = reference.float().reshape(-1,1)
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

    # Training setup
    model = Modified_Uniter_attnbias_rcnn().to(device)
    criterion = FocalLoss(gamma=GAMMA, alpha=ALPHA)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    # Train
    n_iter = 0
    n_prev_iter = 0
    running_loss = 0
    for epoch in range(N_EPOCH):
        for batch_idx, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
 
            input_ids = batch['input_ids'].to(device)
            txt_seg_ids = batch['txt_seg_ids'].to(device)
            vis_feats = batch['vis_feats'].to(device)
            obj_embs = batch['obj_embs'].to(device)
            obj_ids = batch['obj_ids'].to(device)
            pos_x = batch['pos_x'].to(device)
            pos_y = batch['pos_y'].to(device)
            pos_z = batch['pos_z'].to(device)
            bboxes = batch['bboxes'].to(device)
            vis_seg = batch['vis_seg'].to(device)
            extended_attention_mask = batch['extended_attention_mask'].to(device)
            output_mask = batch['output_mask'].to(device)
            reference = batch['reference'].to(device)
            scene_seg = batch['scene_segs'].to(device)
            rel_mask_left = batch['rel_mask_left'].to(device).unsqueeze(0).unsqueeze(2)
            rel_mask_right = batch['rel_mask_right'].to(device).unsqueeze(0).unsqueeze(2)
            rel_mask_up = batch['rel_mask_up'].to(device).unsqueeze(0).unsqueeze(2)
            rel_mask_down = batch['rel_mask_down'].to(device).unsqueeze(0).unsqueeze(2)

            rel_masks = torch.cat((rel_mask_left, rel_mask_right, rel_mask_up, rel_mask_down), axis=0)
            rel_masks = rel_masks * mask_stepper

            truth = reference.float().reshape(-1,1)

            # To fix: NaN under mixed precision
            # with autocast():
                # pred = model(input_ids , txt_seg_ids, vis_feats, obj_embs, obj_ids, pos_x, pos_y, pos_z, bboxes, vis_seg, extended_attention_mask)
                # pred = pred.reshape(1,-1)
                # pred = pred[output_mask==1].reshape(-1,1)

                # loss = criterion(pred, truth)

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            pred = model(input_ids , txt_seg_ids, vis_feats, obj_embs, obj_ids, pos_x, pos_y, pos_z, bboxes, vis_seg, extended_attention_mask, scene_seg, rel_masks)
            pred = pred.reshape(1,-1)
            pred = pred[output_mask==1].reshape(-1,1)
            loss = criterion(pred, truth)
            loss.backward()
            optimizer.step()

            n_iter += 1
            writer.add_scalar('Loss/train_batch', loss, n_iter)
            running_loss += loss.detach()

            if batch_idx % 2000 == 0:
                print(pred.reshape(-1))
                print(truth.reshape(-1))
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
                    os.makedirs(f'./checkpoint/UNITERonCLIPBERT_attnbiasRcnn_n_batchsize{BATCH_SIZE}_lr{LR}_FocalALPHA{ALPHA}_GAMMA{GAMMA}')
                except:
                    pass

                torch.save({
                    'epoch': epoch,
                    'step': n_iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dev_loss': loss,
                    }, f'./checkpoint/UNITERonCLIPBERT_attnbiasRcnn_n_batchsize{BATCH_SIZE}_lr{LR}_FocalALPHA{ALPHA}_GAMMA{GAMMA}/{epoch}_{batch_idx}_{loss}_{f1}.bin')
    print('DONE !!!')

if __name__ == '__main__':
    train()