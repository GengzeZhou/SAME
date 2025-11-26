import argparse
from huggingface_hub import snapshot_download

def main(args):
    if args.data:
        print("Downloading datasets and features from ZGZzz/VersNav...")
        # Download the complete VersNav dataset including all datasets and features
        snapshot_download(
            repo_id="ZGZzz/VersNav",
            repo_type="dataset",
            local_dir="data",
            local_dir_use_symlinks=False
        )
        print("Data download completed!")

    if args.pretrain:
        print("Downloading pretrained models from ZGZzz/SAME...")
        # Download pretrained checkpoints
        if args.model == "attnq" or args.model == "all":
            print("  - Downloading Attn_q pretrained checkpoint...")
            snapshot_download(
                repo_id="ZGZzz/SAME",
                repo_type="model",
                allow_patterns="pretrain/Attnq_pretrained_ckpt.pt",
                local_dir="data",
                local_dir_use_symlinks=False
            )

        if args.model == "attnkv" or args.model == "all":
            print("  - Downloading Attn_kv pretrained checkpoint...")
            snapshot_download(
                repo_id="ZGZzz/SAME",
                repo_type="model",
                allow_patterns="pretrain/Attnkv_pretrained_ckpt.pt",
                local_dir="data",
                local_dir_use_symlinks=False
            )

        if args.model == "ffn" or args.model == "all":
            print("  - Downloading FFN pretrained checkpoint...")
            snapshot_download(
                repo_id="ZGZzz/SAME",
                repo_type="model",
                allow_patterns="pretrain/FFN_pretrained_ckpt.pt",
                local_dir="data",
                local_dir_use_symlinks=False
            )

        print("Pretrained models download completed!")

    if args.checkpoints:
        print("Downloading trained model checkpoints from ZGZzz/SAME...")
        # Download trained model checkpoints
        snapshot_download(
            repo_id="ZGZzz/SAME",
            repo_type="model",
            allow_patterns="ckpt/*",
            local_dir="data",
            local_dir_use_symlinks=False
        )
        print("Trained checkpoints download completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download datasets, pretrained models, and trained checkpoints for SAME"
    )
    parser.add_argument(
        "--data",
        action="store_true",
        default=False,
        help="Download all datasets and features from ZGZzz/VersNav (includes R2R, REVERIE, RXR-EN, CVDN, SOON, OBJNAV_MP3D, and all visual features)"
    )
    parser.add_argument(
        "--pretrain",
        action="store_true",
        default=False,
        help="Download pretrained model weights from ZGZzz/SAME"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["attnq", "attnkv", "ffn", "all"],
        help="Pretrained model type to download: attnq (MoE at Attn_q), attnkv (MoE at Attn_kv), ffn (MoE at FFN), or all"
    )
    parser.add_argument(
        "--checkpoints",
        action="store_true",
        default=False,
        help="Download trained model checkpoints from ZGZzz/SAME"
    )
    args = parser.parse_args()

    # If no arguments provided, show help
    if not (args.data or args.pretrain or args.checkpoints):
        parser.print_help()
        print("\nExample usage:")
        print("  python download.py --data                    # Download all datasets and features")
        print("  python download.py --pretrain --model attnq  # Download Attn_q pretrained model")
        print("  python download.py --pretrain                # Download all pretrained models")
        print("  python download.py --checkpoints             # Download trained checkpoints")
        print("  python download.py --data --pretrain         # Download both data and pretrained models")
    else:
        main(args)
