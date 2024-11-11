import os, random
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.data.dataloader import DataLoader
from pytorch_metric_learning import losses, miners, distances
from torch.utils.tensorboard import SummaryWriter

from utils import util, parser, commons, test
from models import vgl_network
from datasets import gsv_cities, base_dataset

from losses.lotd import LoTD

from itertools import chain

torch.backends.cudnn.benchmark = True  # Provides a speedup
#### Initial setup: parser, logging...
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = os.path.join("logs", args.save_dir, args.backbone_s + "_" + args.aggregation, "gsv_cities", start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(f"Using {torch.cuda.device_count()} GPUs")

#### Creation of Datasets
logging.debug(f"Loading gsv_cities and {args.dataset_name} from folder {args.datasets_folder}")

resize_tmp = args.resize
args.resize = [224, 224]
val_ds = base_dataset.BaseDataset(args, "val")
logging.info(f"Val set: {val_ds}")
args.resize = resize_tmp

#### Initialize teacher model
model_t = vgl_network.VGLNet(args)
model_t = model_t.to("cuda")
util.resume_model(args, model_t)
model_t = torch.nn.DataParallel(model_t)

train_ds = gsv_cities.GSVCitiesDataset(args, cities=(gsv_cities.TRAIN_CITIES))
train_dl = DataLoader(train_ds, batch_size= args.train_batch_size, num_workers=args.num_workers, pin_memory= True)

# lambdas_kd = [100, 1, 0.01, 1]
# teacher_embeddings = []
teacher_feats_1 = []
teacher_feats_2 = []
teacher_feats_3 = []
teacher_feats_4 = []

model_t.eval()
with torch.no_grad():
    for places, labels in tqdm(train_dl, ncols=100, desc="Computing teacher embeddings"):
        BS, N, ch, h, w = places.shape
        images = places.view(BS*N, ch, h, w)

        embeddings_t, feats_t = model_t(images)

        teacher_feats_1.append(feats_t[0].cpu())
        teacher_feats_2.append(feats_t[1].cpu())
        teacher_feats_3.append(feats_t[2].cpu())
        teacher_feats_4.append(feats_t[3].cpu())
        # teacher_embeddings.append(embeddings_t)

teacher_feats_1 = torch.cat(teacher_feats_1)
teacher_feats_2 = torch.cat(teacher_feats_2)
teacher_feats_3 = torch.cat(teacher_feats_3)
teacher_feats_4 = torch.cat(teacher_feats_4)
# teacher_embeddings = torch.cat(teacher_embeddings)

# release GPU Cache
model_t.cpu()
del model_t
torch.cuda.empty_cache()

#### Initialize student model
model_s = vgl_network.MambaVGL(args)
model_s = model_s.to("cuda")
model_s = torch.nn.DataParallel(model_s)

util.print_trainable_parameters(model_s)
# util.print_trainable_layers(model_s)
writer = SummaryWriter('../../tf-logs')

#### Setup Optimizer and Loss
criterion = losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=distances.CosineSimilarity())
criterion_kd1 = LoTD(channel_s = 160)
criterion_kd2 = LoTD(channel_s = 320)
criterion_kd3 = LoTD(channel_s = 640)
criterion_kd4 = LoTD(channel_s = 640)
# criterion_kd = LoRD(num_channels = 640, embedding = True)
optimizer = torch.optim.AdamW(
    chain(model_s.parameters(),
    criterion_kd1.parameters(),
    criterion_kd2.parameters(),
    criterion_kd3.parameters(),
    criterion_kd4.parameters(),
    # criterion_kd.parameters(),
    ), lr=args.lr)

# optimizer = torch.optim.AdamW(model_s.parameters(), lr=args.lr)

scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.2, total_iters=4000)

miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=distances.CosineSimilarity())
if args.use_amp16:
    scaler = torch.cuda.amp.GradScaler()

best_r1 = start_epoch_num = not_improved_num = 0


#### Training loop
for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")

    epoch_start_time = datetime.now()
    epoch_losses=[]
    model_s.train()
    
    for batch_idx, (places, labels) in enumerate(tqdm(train_dl, ncols=100, desc=f"Epoch {epoch_num+1}/{args.epochs_num}")):

        BS, N, ch, h, w = places.shape
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)
        
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            embeddings_s, feats_s = model_s(images)
            
            batch_teacher_feats_1 = teacher_feats_1[batch_idx * BS * N : (batch_idx + 1) * BS * N].cuda()
            batch_teacher_feats_2 = teacher_feats_2[batch_idx * BS * N : (batch_idx + 1) * BS * N].cuda()
            batch_teacher_feats_3 = teacher_feats_3[batch_idx * BS * N : (batch_idx + 1) * BS * N].cuda()
            batch_teacher_feats_4 = teacher_feats_4[batch_idx * BS * N : (batch_idx + 1) * BS * N].cuda()
            # batch_teacher_embeddings = teacher_embeddings[batch_idx * BS * N : (batch_idx + 1) * BS * N]
            
            miner_outputs = miner(embeddings_s, labels)
            loss_ms = criterion(embeddings_s, labels, miner_outputs)
            loss_kd = criterion_kd1(feats_s[0], batch_teacher_feats_1) + criterion_kd2(feats_s[1], batch_teacher_feats_1) + criterion_kd3(feats_s[2], batch_teacher_feats_1) + criterion_kd4(feats_s[3], batch_teacher_feats_1)
            # loss_kd_l2 = criterion_kd(embeddings_s, batch_teacher_embeddings)
            loss = loss_ms + loss_kd # + loss_kd_l2
            # logging.info(f"MS_loss: {loss_ms:.2f}, KD_loss: {loss_kd:.8f}, KD_loss_l2: {loss_kd_l2:.8f}\n")
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
            
        scheduler.step()

        batch_loss = loss.item()
        writer.add_scalar('training loss', batch_loss, epoch_num * len(train_dl) + batch_idx)
        epoch_losses = np.append(epoch_losses, batch_loss)

        del loss, feats_s, miner_outputs, images, labels, batch_loss, loss_ms, embeddings_s
        del embeddings_t, feats_t

    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
        f"average epoch loss = {epoch_losses.mean():.4f}")
    
    # Compute recalls on validation set
    recalls, recalls_str = test.test(args, val_ds, model_s)
    logging.info(f"Recalls on val set {val_ds}: {recalls_str}")
    
    is_best = recalls[0] > best_r1
    
    if is_best:
        logging.info(f"Improved: previous best R@1 = {best_r1:.1f}, current R@1 = {(recalls[0]):.1f}")
        best_r1 = (recalls[0])
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(f"Not improved: {not_improved_num} / {args.patience}: best R@1 = {best_r1:.1f}, current R@1 = {(recalls[0]):.1f}")

    # Save checkpoint, which contains all training parameters
    util.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model_s.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls, "best_r1": best_r1,
        "not_improved_num": not_improved_num
    }, is_best, filename=f"last_model.pth")

    if not_improved_num == args.patience:
        logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
        break

writer.close()
logging.info(f"Best R@1: {best_r1:.1f}")
logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

# del teacher_feats

# update test

args.resize_test_imgs = True
args.resize = [322, 322]
args.resume = f"{args.save_dir}/best_model.pth"

model = vgl_network.MambaVGL(args)
model = model.to("cuda")
model = util.resume_model(args, model)
model = torch.nn.DataParallel(model)

args.dataset_name = "pitts30k"
test_ds = base_dataset.BaseDataset(args, "test")
logging.info(f"Test set: {test_ds}")
recalls, recalls_str = test.test(args, test_ds, model)
logging.info(f"Recalls on {test_ds}: {recalls_str}")

args.dataset_name = "sf_xl"
test_ds = base_dataset.BaseDataset(args, "val")
logging.info(f"Test set: {test_ds}")
recalls, recalls_str = test.test(args, test_ds, model)
logging.info(f"Recalls on {test_ds}: {recalls_str}")