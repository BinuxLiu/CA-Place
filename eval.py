import os
import logging
from datetime import datetime
import torch

from utils import parser, commons, util, test, test_vis
from models import vgl_network, dinov2_network
from datasets import base_dataset


args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = os.path.join("logs", args.save_dir, args.backbone_t + "_" + args.aggregation, args.dataset_name, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

# model = vgl_network.VGLNet_Test(args)
model = vgl_network.MambaVGL(args)
model = model.to("cuda")        

if args.resume != None:
    logging.info(f"Resuming model from {args.resume}")
    model = util.resume_model(args, model)
    
model = torch.nn.DataParallel(model)

test_ds = base_dataset.BaseDataset(args, "test")
logging.info(f"Test set: {test_ds}")
######################################### TEST on TEST SET #########################################
recalls, recalls_str = test.test(args, test_ds, model)

logging.info(f"Recalls on {test_ds}: {recalls_str}")
logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")

