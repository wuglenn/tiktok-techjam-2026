"""Seer CLI - TikTok TechJam 2026.

Sub-commands:
  train   fine-tune the detector        (python main.py train --config ...)
  eval    benchmark a checkpoint        (python main.py eval --checkpoint ...)
  infer   classify an image + heatmap    (python main.py infer --checkpoint ...)
  info    parameter budget report
"""

import sys


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(prog="seer", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="train the detector")
    p_train.add_argument("--config", type=str, default=None)
    p_train.add_argument("--set", nargs="*", default=[], dest="overrides",
                         help="dotted overrides, e.g. max_steps=10 res=224")

    p_eval = sub.add_parser("eval", help="benchmark a checkpoint")
    p_eval.add_argument("--checkpoint", required=True)
    p_eval.add_argument("--dataset", default="comfor_eval")
    p_eval.add_argument("--real-dir", nargs="*", default=None)
    p_eval.add_argument("--fake-dir", nargs="*", default=None)
    p_eval.add_argument("--augmented", action="store_true",
                        help="Pangram augmented protocol (1024px + JPEG q50)")
    p_eval.add_argument("--perturbation", default=None,
                        help="named perturbation, or all (official table) / extra "
                             "(harder NTIRE-style) / all+extra")
    p_eval.add_argument("--hflip-tta", action="store_true")
    p_eval.add_argument("--max-samples", type=int, default=None)
    p_eval.add_argument("--batch-size", type=int, default=16)
    p_eval.add_argument("--out-json", type=str, default=None)
    p_eval.add_argument("--error-dir", type=str, default=None,
                        help="dump the most confident false positives / false "
                             "negatives there as heatmap panels")
    p_eval.add_argument("--error-n", type=int, default=4,
                        help="how many of each to keep (default 4)")

    p_infer = sub.add_parser("infer", help="classify images")
    p_infer.add_argument("--checkpoint", required=True)
    p_infer.add_argument("--image", nargs="+", required=True)
    p_infer.add_argument("--out-dir", default=None)
    p_infer.add_argument("--res", type=int, default=None)

    p_info = sub.add_parser("info", help="parameter budget report")
    p_info.add_argument("--backbone", type=str, default=None)
    p_info.add_argument("--config", type=str, default=None)

    args = parser.parse_args(argv)

    if args.cmd == "train":
        from seer.config import load_config
        from seer.train import run

        cfg = load_config(args.config, args.overrides)
        run(cfg)

    elif args.cmd == "eval":
        from seer.eval import evaluate_checkpoint

        evaluate_checkpoint(
            checkpoint=args.checkpoint,
            dataset=args.dataset,
            augmented=args.augmented,
            perturbation=args.perturbation,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            hflip_tta=args.hflip_tta,
            real_dirs=args.real_dir,
            fake_dirs=args.fake_dir,
            out_json=args.out_json,
            error_dir=args.error_dir,
            error_n=args.error_n,
        )

    elif args.cmd == "infer":
        from seer.infer import infer

        infer(args.checkpoint, args.image, args.out_dir, args.res)

    elif args.cmd == "info":
        from seer.config import load_config
        from seer.model import SeerDetector

        backbone = args.backbone
        probe_layers = None
        if backbone is None:
            cfg = load_config(args.config) if args.config else None
            backbone = cfg.backbone if cfg else "facebook/dinov3-vitl16-pretrain-lvd1689m"
            if cfg and cfg.probe.enabled:
                probe_layers = cfg.probe.layers
        model = SeerDetector(backbone, pretrained=False, probe_layers=probe_layers)
        print(f"backbone: {backbone}")
        if model.probe:
            page = sum(p.numel() for p in model.probe_head.parameters())
            patch = sum(p.numel() for p in model.probe_patch_head.parameters())
            print(
                f"probe   : page + patch linear heads over blocks {model.probe_layers} "
                f"(page {page:,} + patch {patch:,} params)"
            )
        print(f"total  : {model.budget_report()}")


if __name__ == "__main__":
    main()
