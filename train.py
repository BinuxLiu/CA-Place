import os, random
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.data.dataloader import DataLoader
from pytorch_metric_learning import losses, miners, distances
from peft import LoraConfig, get_peft_model
from peft import PeftModel
from torch.utils.tensorboard import SummaryWriter

from utils import util, parser, commons, test
from models import vgl_network, dinov2_network
from datasets import gsv_cities, base_dataset

torch.backends.cudnn.benchmark = True  # Provides a speedup
#### Initial setup: parser, logging...
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = os.path.join("logs", args.save_dir, args.backbone + "_" + args.aggregation, "gsv_cities", start_time.strftime('%Y-%m-%d_%H-%M-%S'))
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

#### Initialize student model
model = vgl_network.MambaVGL(args)
model = model.to("cuda")

if args.aggregation == "netvlad":
    if not args.resume:
        args.dataset_name = "pitts30k"
        cluster_ds = base_dataset.BaseDataset(args, "train")
        model.aggregation.initialize_netvlad_layer(args, cluster_ds, model.backbone)
        del cluster_ds
        
    if args.use_linear:
        args.features_dim = args.clusters * args.linear_dim
        if args.use_cls:
            args.features_dim += 256
    else:
        args.features_dim = args.clusters * dinov2_network.CHANNELS_NUM[args.backbone]

args.resize = resize_tmp

if args.use_lora:
    # model, _, best_r1, start_epoch_num, not_improved_num = util.resume_train(args, model, strict=False)
    trainable_layers = dinov2_network.control_trainable_layer(args.trainable_layers, args.backbone)
    lora_modules = []
    for layer in trainable_layers:
        lora_modules += [
            f"{layer}.attn.q", f"{layer}.attn.k", f"{layer}.attn.v", f"{layer}.attn.proj",
            f"{layer}.mlp.fc1", f"{layer}.mlp.fc2"]
    lora_config = LoraConfig(r=64, lora_alpha=128, use_dora= False, target_modules=lora_modules, lora_dropout=0.01, modules_to_save=["aggregation"])
    model = get_peft_model(model, lora_config)

if args.aggregation == "netvlad" and args.use_linear and args.resume != None:
    for name, param in model.named_parameters():
        if "aggregation.feat_proj" in name or "aggregation.cls_proj" in name :
            param.requires_grad = True
        else:
            param.requires_grad = False
    logging.info(f"Linear as Learned PCA and only fine-tuning linear.")

model = torch.nn.DataParallel(model)
    
util.print_trainable_parameters(model)
util.print_trainable_layers(model)
writer = SummaryWriter('../../tf-logs')

#### Setup Optimizer and Loss
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
if not args.resume:
    if not args.use_extra_datasets:
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.2, total_iters=4000)
    else:
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.2, total_iters=15000)
criterion = losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0, distance=distances.CosineSimilarity())
miner = miners.MultiSimilarityMiner(epsilon=0.1, distance=distances.CosineSimilarity())
if args.use_amp16:
    scaler = torch.cuda.amp.GradScaler()

#### Resume model, optimizer, and other training parameters
if args.resume:
    model, _, best_r1, start_epoch_num, not_improved_num = util.resume_train(args, model, strict=False)
    logging.info(f"Resuming from epoch {start_epoch_num} with best recall@1 {best_r1:.1f}")
    if args.aggregation == "netvlad" and args.use_linear:
        best_r1 = 0
else:
    best_r1 = start_epoch_num = not_improved_num = 0


#### Training loop
for epoch_num in range(start_epoch_num, args.epochs_num):
    logging.info(f"Start training epoch: {epoch_num:02d}")

    random_datasets = gsv_cities.TRAIN_CITIES.copy()
    random.shuffle(random_datasets)
    train_ds = gsv_cities.GSVCitiesDataset(args, cities=(random_datasets))
    train_dl = DataLoader(train_ds, batch_size= args.train_batch_size, num_workers=args.num_workers, pin_memory= True)
    epoch_start_time = datetime.now()
    epoch_losses=[]

    model.train()
    for batch_idx, (places, labels) in enumerate(tqdm(train_dl, ncols=100, desc=f"Epoch {epoch_num+1}/{args.epochs_num}")):

        BS, N, ch, h, w = places.shape
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)
        
        optimizer.zero_grad()

        if not args.use_amp16:
            features = model(images)
            miner_outputs = miner(features, labels)
            loss = criterion(features, labels, miner_outputs)
            loss.backward()
            optimizer.step()
        else:
            with torch.cuda.amp.autocast():
                features = model(images)
                miner_outputs = miner(features, labels)
                loss = criterion(features, labels, miner_outputs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        if not args.resume:
            scheduler.step()

        batch_loss = loss.item()
        writer.add_scalar('training loss', batch_loss, epoch_num * len(train_dl) + batch_idx)
        epoch_losses = np.append(epoch_losses, batch_loss)

        del loss, features, miner_outputs, images, labels, batch_loss

    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
        f"average epoch loss = {epoch_losses.mean():.4f}")
    
    # Compute recalls on validation set
    recalls, recalls_str = test.test(args, val_ds, model)
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
    util.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls, "best_r1": best_r1,
        "not_improved_num": not_improved_num
    }, is_best, filename=f"last_model.pth")

    if args.use_lora:
        model.module.save_pretrained(os.path.join(args.save_dir, "lora"))

    if not_improved_num == args.patience and not args.resume:
        logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
        break

writer.close()
logging.info(f"Best R@1: {best_r1:.1f}")
logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

# update test
args.dataset_name = "pitts30k"
args.resize_test_imgs = False
args.resume = f"{args.save_dir}/best_model.pth"

model = vgl_network.MambaVGL(args)
model = model.to("cuda")

if args.use_lora:
    model = PeftModel.from_pretrained(model, f"{args.save_dir}/lora")
else:
    model = util.resume_model(args, model)

model = torch.nn.DataParallel(model)

test_ds = base_dataset.BaseDataset(args, "test")
logging.info(f"Test set: {test_ds}")
recalls, recalls_str = test.test(args, test_ds, model)
logging.info(f"Recalls on {test_ds}: {recalls_str}")

logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")

args.dataset_name = "sf_xl"

test_ds = base_dataset.BaseDataset(args, "val")
logging.info(f"Test set: {test_ds}")
recalls, recalls_str = test.test(args, test_ds, model)
logging.info(f"Recalls on {test_ds}: {recalls_str}")

logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")