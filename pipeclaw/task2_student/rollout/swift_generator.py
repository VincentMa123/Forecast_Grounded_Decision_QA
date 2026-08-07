"""Model loading and generation for autonomous rollouts.

``SwiftGenerator`` is the only place in the rollout package that imports PEFT
or MS-SWIFT and owns model weights.  The suite only imports torch lazily for
between-case allocator cleanup, keeping dry runs hardware-free.

Base-model quantization is resolved from the adapter checkpoint's own
``args.json`` so evaluation loads the same weight representation that training
used.  ``--quant-bits`` overrides it explicitly and ``--no-quantization``
restores full-precision loading.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


def read_saved_training_args(adapter_dir: Path) -> dict[str, Any]:
    """Read MS-SWIFT's saved model-loading arguments when available."""

    candidates = (adapter_dir / "args.json", adapter_dir.parent / "args.json")
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def normalize_dtype_name(value: Any) -> str | None:
    """Return a bare torch dtype name such as ``bfloat16``."""

    if value is None:
        return None
    name = str(value).strip()
    if name.startswith("torch."):
        name = name[len("torch.") :]
    return name or None


def normalize_quant_bits(value: Any) -> int | None:
    """Return 4 or 8, or None when the value is absent or unsupported."""

    try:
        bits = int(value)
    except (TypeError, ValueError):
        return None
    return bits if bits in {4, 8} else None


def normalize_bool(value: Any, *, default: bool) -> bool:
    """Interpret saved JSON/YAML boolean spellings."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return default


def resolve_model_load_kwargs(
    adapter_dir: Path,
    *,
    quant_bits: int | None = None,
    no_quantization: bool = False,
) -> dict[str, Any]:
    """Resolve base-model quantization from CLI overrides or checkpoint metadata.

    MS-SWIFT stores the QLoRA loading arguments in ``args.json`` next to a
    checkpoint, but the custom Python evaluator does not load that file on its
    own.  Reusing it keeps evaluation on the same base-weight representation as
    training.  An explicit ``quant_bits`` override always uses bitsandbytes;
    ``no_quantization`` deliberately restores the old full-precision behavior.
    """

    if no_quantization:
        return {}

    saved_args = read_saved_training_args(adapter_dir)
    explicit_override = quant_bits is not None
    bits = normalize_quant_bits(
        quant_bits if explicit_override else saved_args.get("quant_bits")
    )
    if bits is None:
        return {}

    method = (
        "bnb"
        if explicit_override
        else str(saved_args.get("quant_method") or "").strip().lower()
    )
    if not method:
        method = "bnb"
    if method != "bnb":
        raise ValueError(
            f"Unsupported checkpoint quant_method {method!r}; "
            "use a bitsandbytes checkpoint or pass --no-quantization"
        )

    kwargs: dict[str, Any] = {
        "quant_method": method,
        "quant_bits": bits,
        "torch_dtype": normalize_dtype_name(saved_args.get("torch_dtype")) or "bfloat16",
    }
    if bits == 4:
        kwargs.update(
            {
                "bnb_4bit_compute_dtype": (
                    normalize_dtype_name(saved_args.get("bnb_4bit_compute_dtype"))
                    or "bfloat16"
                ),
                "bnb_4bit_quant_type": str(
                    saved_args.get("bnb_4bit_quant_type") or "nf4"
                ),
                "bnb_4bit_use_double_quant": normalize_bool(
                    saved_args.get("bnb_4bit_use_double_quant"), default=True
                ),
            }
        )
        quant_storage = normalize_dtype_name(saved_args.get("bnb_4bit_quant_storage"))
        if quant_storage:
            kwargs["bnb_4bit_quant_storage"] = quant_storage
    return kwargs


def build_model_load_kwargs(model_load_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Build the Transformers kwargs required for actual BNB quantized loading."""

    if not model_load_spec:
        return {}

    bits = normalize_quant_bits(model_load_spec.get("quant_bits"))
    if bits is None or model_load_spec.get("quant_method") != "bnb":
        return {}

    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - MS-SWIFT installs transformers
        raise RuntimeError(
            "bitsandbytes quantization requires Transformers.BitsAndBytesConfig"
        ) from exc

    config_kwargs: dict[str, Any] = {
        "load_in_4bit": bits == 4,
        "load_in_8bit": bits == 8,
    }
    if bits == 4:
        config_kwargs.update(
            {
                "bnb_4bit_compute_dtype": model_load_spec.get(
                    "bnb_4bit_compute_dtype", "bfloat16"
                ),
                "bnb_4bit_quant_type": model_load_spec.get("bnb_4bit_quant_type", "nf4"),
                "bnb_4bit_use_double_quant": model_load_spec.get(
                    "bnb_4bit_use_double_quant", True
                ),
            }
        )
        if model_load_spec.get("bnb_4bit_quant_storage") is not None:
            config_kwargs["bnb_4bit_quant_storage"] = model_load_spec[
                "bnb_4bit_quant_storage"
            ]

    return {
        "torch_dtype": model_load_spec.get("torch_dtype", "bfloat16"),
        "quantization_config": BitsAndBytesConfig(**config_kwargs),
    }


def coerce_torch_dtype(model_load_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Convert saved dtype names to torch dtypes for the Python MS-SWIFT API."""

    kwargs = dict(model_load_kwargs)
    try:
        import torch

        for key in ("torch_dtype", "bnb_4bit_compute_dtype", "bnb_4bit_quant_storage"):
            dtype_name = kwargs.get(key)
            if isinstance(dtype_name, str):
                kwargs[key] = getattr(torch, dtype_name)
    except ImportError:  # pragma: no cover - CUDA env supplies torch
        return kwargs
    except AttributeError:
        # Leave an unknown dtype untouched so MS-SWIFT can report its own
        # version-specific validation error rather than failing here.
        pass
    return kwargs


def discover_base_model(adapter_dir: Path) -> str:
    """Infer the base model from the adapter checkpoint's saved metadata."""

    config_path = adapter_dir / "adapter_config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        model = config.get("base_model_name_or_path")
        if model:
            return str(model)
    saved_args = read_saved_training_args(adapter_dir)
    model = saved_args.get("model")
    if model:
        return str(model)
    raise ValueError(
        "--model is required when the adapter has no base model in "
        "adapter_config.json or args.json"
    )


class SwiftGenerator:
    """Small adapter around MS-SWIFT's TransformersEngine."""

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    @classmethod
    def from_args(
        cls,
        *,
        model: str,
        adapters: str,
        device: str | None = None,
        quant_bits: int | None = None,
        no_quantization: bool = False,
    ) -> "SwiftGenerator":
        if device:
            os.environ["CUDA_VISIBLE_DEVICES"] = device
        try:
            from peft import PeftModel
            from swift import get_model_processor, get_template
            from swift.infer_engine import TransformersEngine
        except ImportError as exc:  # pragma: no cover - depends on the training env
            raise RuntimeError(
                "MS-SWIFT and PEFT are required for non-dry-run evaluation"
            ) from exc

        model_load_spec = resolve_model_load_kwargs(
            Path(adapters),
            quant_bits=quant_bits,
            no_quantization=no_quantization,
        )
        model_load_kwargs = coerce_torch_dtype(build_model_load_kwargs(model_load_spec))
        if model_load_spec:
            print(
                "[evaluate_autonomous] base-model loading: "
                f"quant_method={model_load_spec.get('quant_method')!r}, "
                f"quant_bits={model_load_spec.get('quant_bits')!r}"
            )
        else:
            print("[evaluate_autonomous] base-model loading: default (unquantized)")

        model_obj, processor = get_model_processor(model, **model_load_kwargs)
        model_obj = PeftModel.from_pretrained(model_obj, adapters)
        template = get_template(processor, enable_thinking=False)
        engine = TransformersEngine(model_obj, template=template)
        return cls(engine)

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> Any:
        from swift.infer_engine import InferRequest, RequestConfig

        request = InferRequest(
            messages=[dict(message) for message in messages],
            tools=list(tools) or None,
        )
        config = RequestConfig(
            max_tokens=max_tokens, temperature=temperature, stream=False
        )
        responses = self.engine.infer([request], request_config=config)
        if not responses:
            return ""
        return responses[0]
