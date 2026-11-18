
import argparse
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from laya.agent import _fix_tokenizer_config
from laya.train import TrainConfig, finetune

MODEL_ID = "convaiinnovations/laya"
MODEL_DIR = Path("./laya_base")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("rows")
    p.add_argument("output_dir")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--calib-max", type=int, default=400)
    p.add_argument("--no-checkpointing", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available - fix your torch install before training.")
    print("GPU:", torch.cuda.get_device_name(0))

    if not (MODEL_DIR / "model.safetensors").exists():
        snapshot_download(MODEL_ID, local_dir=str(MODEL_DIR))
    _fix_tokenizer_config(str(MODEL_DIR))

    config = TrainConfig(
        epochs=args.epochs,
        micro_batch=args.micro_batch,
        grad_accum=args.grad_accum,
        calib_max=args.calib_max,
        calib_seed=20260922,
        seed=42,
        max_len=1024,
        head_max_len=256,
        gradient_checkpointing=not args.no_checkpointing,
        amp=False,
    )

    summary = finetune(
        data=args.rows,
        model_dir=str(MODEL_DIR),
        output_dir=args.output_dir,
        config=config,
        device="cuda",
    )

    print(f"Train items: {summary['train_items']}; calibration items: {summary['calibration_items']}")
    print("Fitted temperatures (choice, score, noul):", [round(t, 3) for t in summary["temperature"]])
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
