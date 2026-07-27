"""Train the mock lifecycle checkpoint with intervention-focused supervision."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mock_decoder.json")
    parser.add_argument("--resume-from-checkpoint")
    return parser.parse_args()


def _load_config(path: Path):
    from training.config import TrainingConfig

    payload = json.loads(path.read_text(encoding="utf-8"))
    causal = dict(payload.pop("causal_training", {}))
    if not causal:
        raise ValueError("Training config must define causal_training settings.")
    return TrainingConfig.from_dict(payload), causal


def _configure_device(config_path: Path) -> None:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    device = payload.get("device", "auto")
    if isinstance(device, int) and device >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    elif isinstance(device, str) and (device.isdigit() or device.startswith("cuda:")):
        os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[-1]


def main() -> int:
    args = _parse_args()
    config_path = Path(args.config).resolve()
    _configure_device(config_path)

    from training.causal import CausalFluidTrainer, CausalWindowDataset, load_intervention_manifest
    from training.utils import run_training, setup_training

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    config, causal = _load_config(config_path)
    setup = setup_training(config)
    manifest = load_intervention_manifest(PROJECT_ROOT / causal["manifest_path"])
    wrapped_train = CausalWindowDataset(
        setup["train_dataset"],
        manifest,
        window_repeat=int(causal.get("window_repeat", 4)),
        window_before=int(causal.get("window_before", 4)),
        window_after=int(causal.get("window_after", 12)),
    )
    trainer = CausalFluidTrainer(
        model=setup["model"],
        args=setup["trainer"].args,
        training_config=config,
        train_dataset=wrapped_train,
        eval_dataset=setup["val_dataset"],
        normalizer=None,
        tokenizer=setup["tokenizer"],
        intervention_manifest=manifest,
        variable_to_index=setup["train_dataset"].variable_to_index,
        causal_auxiliary_loss_weight=float(causal.get("auxiliary_loss_weight", 4.0)),
        causal_post_intervention_steps=causal.get("post_intervention_steps", 30),
    )
    logging.getLogger(__name__).info(
        "Causal training examples: base=%d repeated=%d",
        len(setup["train_dataset"]),
        len(wrapped_train),
    )
    results = run_training(trainer, resume_from_checkpoint=args.resume_from_checkpoint)
    output = Path(config.output_dir) / "training_results.json"
    output.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
