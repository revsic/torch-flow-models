import argparse
import json
import re
from pathlib import Path
from uuid import uuid4

import torch
import safetensors
from safetensors.torch import load_model

from fid import compute_fid_with_model
from flowmodels.sct import ScaledContinuousCM, ScaledContinuousCMScheduler, TrigFlow
from ddpmpp import DDPMpp
from trainer import _LossDDPWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", choices=["sct", "trigflow"], default="sct")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--ref", default="./cifar10-32x32.npz")
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--dump", default="/data0/images")
    args = parser.parse_args()

    backbone = DDPMpp(
        resolution=32,
        in_channels=3,
        nf=128,
        ch_mult=[1, 2, 2, 2],
        attn_resolutions=[16],
        num_res_blocks=4,
        init_scale=0.0,
        skip_rescale=True,
        dropout=0.20,
        pe_scale=0.02,
        use_shift_scale_norm=True,
        use_double_norm=True,
    )

    match args.frame:
        case "sct":
            model = ScaledContinuousCM(backbone, ScaledContinuousCMScheduler())
        case "trigflow":
            model = TrigFlow(backbone, ScaledContinuousCMScheduler())
        case _:
            assert False, "invalid framework"

    if not (ckpt := Path(args.ckpt)).exists():
        print("[*] CHECKPOINT DOES NOT EXIST: ", args.ckpt)
        exit(1)

    found = re.findall(
        r"^(.+?)/(2025\.\d{2}\.\d{2}KST\d{2}:\d{2}:\d{2})/ckpt/(\d+)/(.+?)\.safetensors$",
        ckpt.absolute().as_posix(),
    )
    dump: Path
    if not found:
        dump = Path(args.dump) / f"{uuid4().hex}"
    else:
        (parent, date, steps, head), *_ = found
        dump = (
            Path(args.dump)
            / f"{Path(parent).stem}.{date}.ckpt{steps}.steps{args.steps}.{head}"
        )

    _enable_loss_ddp_wrapper = False
    with safetensors.safe_open(args.ckpt, framework="pt") as f:
        key = next(iter(f.keys()))
        if key.startswith("model."):
            _enable_loss_ddp_wrapper = True

    if _enable_loss_ddp_wrapper:
        load_model(_LossDDPWrapper(model), args.ckpt)
    else:
        print("[*] WARNING: _LossDDPWrapper IS NOT ENABLED")
        load_model(model, args.ckpt)

    model.to("cuda:0")
    model.eval()

    with open(f"./test.{dump.stem}.json", "w") as f:
        json.dump(
            log := {
                "frame": args.frame,
                "ckpt": ckpt.absolute().as_posix(),
                "dump": dump.absolute().as_posix(),
                "steps": args.steps,
                "ref": args.ref,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    dump.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        fid = compute_fid_with_model(
            model,
            steps=args.steps,
            num_samples=args.samples,
            sampling_batch_size=1024,
            inception_batch_size=1024,
            device="cuda:0",
            cache=args.ref,
            scaler=lambda x: ((x - x.amin()) / (x.amax() - x.amin()) * 255).to(
                torch.uint8
            ),
            _save_images=dump,
        )

    log["fid"] = fid
    with open(f"./test.{dump.stem}.json", "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(log)


if __name__ == "__main__":
    main()
