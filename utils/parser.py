import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Cross-Arch VPR",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Ablation parameters
    parser.add_argument("--resize_test_imgs", action="store_true",
                        help="traing or testing")
    parser.add_argument("--num_hiddens", type=int, default=3,
                        help="channel attention")
    parser.add_argument("--lambda_kd", type=int, default=1,
                        help="")
    parser.add_argument("--ca_method", type=str, default="gem",
                        choices=["gem", "avg", "cba"], help="_")
    parser.add_argument("--use_cls", action="store_true",
                        help="use cls tokens of DINOv2")
    parser.add_argument("--use_ca", action="store_true",
                        help="use channel attention modules")
    parser.add_argument("--efficient_ram_testing", action="store_true",
                        help="this testing method designed for large-scale datasets")
    parser.add_argument('--w_dist', type=float, default=25.0, help='weight for RKD distance')
    parser.add_argument('--w_angle', type=float, default=50.0, help='weight for RKD angle')
    # Training parameters
    parser.add_argument("--use_amp16", action="store_true",
                        help="use Automatic Mixed Precision")
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Batch size for training.")
    parser.add_argument("--infer_batch_size", type=int, default=64,
                        help="Batch size for inference.")
    parser.add_argument("--epochs_num", type=int, default=4,
                        help="number of epochs to train")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.00006, help="_")
    # Model parameters
    parser.add_argument("--backbone_t", type=str, default="dinov2_vitb14",
                        choices=["dinov2_vitb14", "dinov2_vits14", "dinov2_vitl14", "dinov2_vitg14"], help="_")
    parser.add_argument("--backbone_s", type=str, default="mambavision_t",
                        choices=["mambavision_t", "mambavision_t2"], help="_")
    parser.add_argument("--aggregation", type=str, default="g2m", choices=["salad", "cosgem", "cls", "g2m"])
    parser.add_argument("--trainable_layers", type=str, default="8, 9, 10, 11",
                    help="Comma-separated list of layer indexes to be trained")
    parser.add_argument("--features_dim", type=int, default=768,
                        help="features_dim")
    parser.add_argument("--clusters", type=int, default=64)
    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")
    # Other parameters
    parser.add_argument("--num_workers", type=int, default=16, help="num_workers for all dataloaders")
    parser.add_argument('--resize', type=int, default=[224, 224], nargs=2, help="Resizing shape for images (HxW).")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25, help="_")
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 100], nargs="+",
                        help="Recalls to be computed, such as R@1.")
    # Paths parameters
    parser.add_argument("--datasets_folder", type=str, default="/root/autodl-tmp", help="Path with all datasets")
    # parser.add_argument("--datasets_folder", type=str, default="/media/hello/data1/binux/datasets", help="Path with all datasets")
    # parser.add_argument("--datasets_folder", type=str, default="/mnt/sda3/Projects/npr/datasets", help="Path with all datasets")
    parser.add_argument("--dataset_name", type=str, default="pitts30k", help="Relative path of the dataset")
    parser.add_argument("--queries_name", type=str, default=None,
                        help="Path with images to be queried")
    parser.add_argument("--save_dir", type=str, default="",
                        help="Folder name of the current run (saved in ./logs/)")
    parser.add_argument("--num_preds_to_save", type=int, default=0,
                        help="set != 0 if you want to save predictions for each query")
    parser.add_argument("--save_only_wrong_preds", action="store_true",
                        help="set to true if you want to save predictions only for "
                        "wrongly predicted queries")
    
    args = parser.parse_args()

    return args