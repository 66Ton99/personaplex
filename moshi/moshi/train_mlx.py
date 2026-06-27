"""Memory-efficient LoRA training for PersonaPlex on Apple Silicon."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.nn.utils import checkpoint
from mlx.utils import tree_flatten, tree_map, tree_unflatten
import numpy as np
from moshi_mlx import models
from moshi_mlx.models.lm import DepFormerConfig, LmConfig
from moshi_mlx.modules.transformer import TransformerConfig


GB = 1024**3
TEXT_PADDING_IDS = (0, 3)
ZERO_TOKEN_ID = -1


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        output_dims, input_dims = base.weight.shape
        self.base = base
        self.lora_a = mx.random.normal((rank, input_dims)) * 0.01
        self.lora_b = mx.zeros((output_dims, rank))
        self.scale = alpha / rank

    def __call__(self, inputs: mx.array) -> mx.array:
        base = self.base(inputs)
        delta = (inputs @ self.lora_a.T) @ self.lora_b.T
        return base + (delta * self.scale).astype(base.dtype)

    def fuse(self) -> nn.Linear:
        output_dims, input_dims = self.base.weight.shape
        fused = nn.Linear(input_dims, output_dims, bias="bias" in self.base)
        delta = (self.lora_b @ self.lora_a) * self.scale
        fused.weight = self.base.weight + delta.astype(self.base.weight.dtype)
        if "bias" in self.base:
            fused.bias = self.base.bias
        mx.eval(fused.parameters())
        return fused


class LoRAEmbedding(nn.Module):
    def __init__(self, base: nn.Module, rank: int, alpha: float):
        super().__init__()
        vocab_size, embedding_dim = base.weight.shape
        self.base = base
        self.lora_a = mx.random.normal((rank, embedding_dim)) * 0.01
        self.lora_b = mx.zeros((vocab_size, rank))
        self.scale = alpha / rank

    def __call__(self, token_ids: mx.array) -> mx.array:
        base = self.base(token_ids)
        safe_ids = mx.maximum(token_ids, 0)
        delta = self.lora_b[safe_ids] @ self.lora_a
        delta = mx.where((token_ids < 0)[..., None], mx.zeros_like(delta), delta)
        return base + (delta * self.scale).astype(base.dtype)

    def fuse(self) -> nn.Module:
        delta = (self.lora_b @ self.lora_a) * self.scale
        self.base.weight = self.base.weight + delta.astype(self.base.weight.dtype)
        mx.eval(self.base.weight)
        return self.base


def _adapt_linear(parent: nn.Module, name: str, rank: int, alpha: float) -> None:
    layer = getattr(parent, name)
    if not isinstance(layer, LoRALinear):
        setattr(parent, name, LoRALinear(layer, rank, alpha))


def inject_lora(
    model: models.Lm,
    rank: int,
    alpha: float,
    train_layers: int,
    adapt_input_embedding: bool = False,
) -> None:
    if train_layers > len(model.transformer.layers):
        raise ValueError(
            f"cannot train {train_layers} layers; model has {len(model.transformer.layers)}"
        )
    model.freeze()
    layers = model.transformer.layers[-train_layers:]
    for layer in layers:
        _adapt_linear(layer.self_attn, "in_proj", rank, alpha)
        _adapt_linear(layer.self_attn, "out_proj", rank, alpha)
        _adapt_linear(layer.gating, "linear_in", rank, alpha)
        _adapt_linear(layer.gating, "linear_out", rank, alpha)
    if adapt_input_embedding:
        model.text_emb = LoRAEmbedding(model.text_emb, rank, alpha)
    _adapt_linear(model, "text_linear", rank, alpha)
    model._lora_train_start = len(model.transformer.layers) - train_layers
    model._adapt_input_embedding = adapt_input_embedding

    for slice_ in model.depformer.slices:
        _adapt_linear(slice_, "linear_in", rank, alpha)
        _adapt_linear(slice_, "linear_out", rank, alpha)
    model.depformer.slices[0].emb = LoRAEmbedding(
        model.depformer.slices[0].emb,
        rank,
        alpha,
    )


def _attention(
    attention: nn.Module,
    inputs: mx.array,
    mask: mx.array | None,
    cache: tuple[mx.array, mx.array] | None = None,
) -> tuple[mx.array, tuple[mx.array, mx.array]]:
    batch, steps, model_dim = inputs.shape
    config = attention.cfg
    qkv = attention.in_proj(inputs).reshape(
        batch,
        steps,
        3,
        config.num_heads,
        config.head_dim,
    )
    query = qkv[:, :, 0].transpose(0, 2, 1, 3)
    key = qkv[:, :, 1].transpose(0, 2, 1, 3)
    value = qkv[:, :, 2].transpose(0, 2, 1, 3)
    offset = 0 if cache is None else cache[0].shape[2]
    if attention.rope is not None:
        query = attention.rope(query, offset=offset)
        key = attention.rope(key, offset=offset)
    if cache is not None:
        key = mx.concatenate([cache[0], key], axis=2)
        value = mx.concatenate([cache[1], value], axis=2)
    output = mx.fast.scaled_dot_product_attention(
        query,
        key,
        value,
        scale=attention.scale,
        mask=mask,
    )
    output = output.transpose(0, 2, 1, 3).reshape(batch, steps, model_dim)
    return attention.out_proj(output), (key, value)


def _main_layer(layer: nn.Module, inputs: mx.array, mask: mx.array) -> mx.array:
    residual = inputs
    attention, _ = _attention(layer.self_attn, layer.norm1(inputs), mask)
    inputs = residual + layer.layer_scale_1(attention)
    if layer.cross_attention is not None:
        raise ValueError("cross-attention training is not supported")
    return inputs + layer.layer_scale_2(layer.gating(layer.norm2(inputs)))


def _main_transformer(
    transformer: nn.Module,
    inputs: mx.array,
    gradient_checkpointing: bool,
    train_start: int,
    stop_prefix_gradient: bool,
) -> mx.array:
    steps = inputs.shape[1]
    mask = mx.triu(mx.full((steps, steps), -1e9, dtype=inputs.dtype), k=1)
    for layer_index, layer in enumerate(transformer.layers):
        if stop_prefix_gradient and layer_index == train_start:
            inputs = mx.stop_gradient(inputs)
        if gradient_checkpointing:
            forward = checkpoint(
                layer,
                lambda values, current=layer: _main_layer(current, values, mask),
            )
            inputs = forward(inputs)
        else:
            inputs = _main_layer(layer, inputs, mask)
    return inputs


def _depth_layer(
    layer: nn.Module,
    inputs: mx.array,
    cache: tuple[mx.array, mx.array] | None,
) -> tuple[mx.array, tuple[mx.array, mx.array]]:
    residual = inputs
    attention, cache = _attention(layer.self_attn, layer.norm1(inputs), None, cache)
    inputs = residual + layer.layer_scale_1(attention)
    inputs = inputs + layer.layer_scale_2(layer.gating(layer.norm2(inputs)))
    return inputs, cache


def _delay_codes(codes: mx.array, delays: list[int], initial: mx.array) -> mx.array:
    _, streams, steps = codes.shape
    delayed = []
    for stream in range(streams):
        delay = delays[stream]
        if delay == 0:
            delayed.append(codes[:, stream])
        else:
            prefix = mx.broadcast_to(initial[:, stream], (codes.shape[0], delay))
            delayed.append(mx.concatenate([prefix, codes[:, stream, : steps - delay]], axis=1))
    return mx.stack(delayed, axis=1)


def _cross_entropy_sum(
    logits: mx.array,
    targets: mx.array,
    weights: mx.array,
) -> tuple[mx.array, mx.array]:
    safe_targets = mx.maximum(targets, 0)
    losses = nn.losses.cross_entropy(
        logits.astype(mx.float32),
        safe_targets,
        reduction="none",
    )
    return mx.sum(losses * weights), mx.sum(weights)


def training_loss(
    model: models.Lm,
    codes: mx.array,
    gradient_checkpointing: bool = True,
    first_codebook_weight: float = 100.0,
    text_padding_weight: float = 0.3,
) -> mx.array:
    batch, streams, steps = codes.shape
    if streams != model.cfg.audio_codebooks + 1:
        raise ValueError(f"expected {model.cfg.audio_codebooks + 1} streams, got {streams}")
    delays = [0, *model.cfg.audio_delays]
    initial = mx.concatenate(
        [
            mx.full((batch, 1, 1), model.cfg.text_out_vocab_size, dtype=mx.int32),
            mx.full(
                (batch, model.cfg.audio_codebooks, 1),
                model.cfg.audio_vocab_size - 1,
                dtype=mx.int32,
            ),
        ],
        axis=1,
    )
    delayed = _delay_codes(codes, delays, initial)
    delayed = mx.concatenate([initial, delayed], axis=2)
    inputs = delayed[:, :, :-1]
    depformer_targets = delayed[:, :, 1:]

    hidden = model.text_emb(inputs[:, 0])
    for stream, embedding in enumerate(model.audio_embs):
        hidden = hidden + embedding(inputs[:, stream + 1])
    hidden = _main_transformer(
        model.transformer,
        hidden,
        gradient_checkpointing=gradient_checkpointing,
        train_start=model._lora_train_start,
        stop_prefix_gradient=not model._adapt_input_embedding,
    )
    hidden = model.out_norm(hidden)
    text_logits = model.text_linear(hidden)

    text_targets = codes[:, 0]
    text_weights = (text_targets != ZERO_TOKEN_ID).astype(mx.float32)
    for padding_id in TEXT_PADDING_IDS:
        text_weights = mx.where(
            text_targets == padding_id,
            text_weights * text_padding_weight,
            text_weights,
        )
    text_sum, text_weight = _cross_entropy_sum(
        text_logits,
        text_targets,
        text_weights,
    )

    depth_cache: list[tuple[mx.array, mx.array] | None] = [
        None for _ in model.depformer.slices[0].transformer.layers
    ]
    audio_sum = mx.array(0.0)
    audio_weight = mx.array(0.0)
    previous_tokens = depformer_targets[:, 0]
    for codebook, slice_ in enumerate(model.depformer.slices):
        depth = slice_.linear_in(hidden) + slice_.emb(previous_tokens)
        depth = depth.reshape(batch * steps, 1, -1)
        next_cache = []
        for layer_index, layer in enumerate(slice_.transformer.layers):
            depth, layer_cache = _depth_layer(
                layer,
                depth,
                depth_cache[layer_index],
            )
            next_cache.append(layer_cache)
        depth_cache = next_cache
        logits = slice_.linear_out(depth[:, 0]).reshape(batch, steps, -1)

        delay = model.cfg.audio_delays[codebook]
        valid_steps = steps - delay
        targets = codes[:, codebook + 1, :valid_steps]
        aligned_logits = logits[:, delay : delay + valid_steps]
        weights = (targets != ZERO_TOKEN_ID).astype(mx.float32)
        if codebook == 0:
            weights = weights * first_codebook_weight
        loss_sum, weight_sum = _cross_entropy_sum(aligned_logits, targets, weights)
        audio_sum = audio_sum + loss_sum
        audio_weight = audio_weight + weight_sum
        previous_tokens = depformer_targets[:, codebook + 1]

    return text_sum / mx.maximum(text_weight, 1.0) + audio_sum / mx.maximum(audio_weight, 1.0)


class TokenDataset:
    def __init__(self, manifest: Path, sequence_frames: int, seed: int, min_text_tokens: int):
        self.sequence_frames = sequence_frames
        self.min_text_tokens = min_text_tokens
        self.seed = seed
        self.random = random.Random(seed)
        with manifest.open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        if not self.rows:
            raise ValueError(f"empty dataset: {manifest}")

    def reset(self) -> None:
        self.random.seed(self.seed)

    def sample(self) -> mx.array:
        for _ in range(32):
            row = self.random.choice(self.rows)
            with np.load(row["path"]) as archive:
                codes = archive["codes"]
            if codes.shape[1] < self.sequence_frames:
                continue
            start = self.random.randint(0, codes.shape[1] - self.sequence_frames)
            chunk = codes[:, start : start + self.sequence_frames]
            text = chunk[0]
            real_text = np.count_nonzero(
                (text != ZERO_TOKEN_ID) & (text != 0) & (text != 3)
            )
            if real_text >= self.min_text_tokens:
                return mx.array(chunk[None], dtype=mx.int32)
        raise RuntimeError("could not sample a chunk with enough assistant text tokens")


def _trainable_count(model: nn.Module) -> int:
    return sum(value.size for _, value in tree_flatten(model.trainable_parameters()))


def _clip_gradients(gradients: dict[str, Any], max_norm: float) -> dict[str, Any]:
    squared = mx.array(0.0)
    for _, gradient in tree_flatten(gradients):
        squared = squared + mx.sum(gradient.astype(mx.float32) ** 2)
    norm = mx.sqrt(squared)
    scale = mx.minimum(1.0, max_norm / (norm + 1e-6))
    return tree_map(lambda gradient: gradient * scale, gradients)


def _save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    output_dir: Path,
    step: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"adapter-{step:06d}.safetensors"
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(path), weights, metadata={"step": str(step)})
    optimizer_path = output_dir / f"adapter-{step:06d}.optimizer.safetensors"
    optimizer_weights = dict(tree_flatten(optimizer.state))
    mx.save_safetensors(
        str(optimizer_path),
        optimizer_weights,
        metadata={"step": str(step)},
    )


def _resume_optimizer(
    optimizer: optim.Optimizer,
    model: nn.Module,
    adapter_path: Path | None,
    reset_optimizer: bool,
) -> int:
    if adapter_path is None:
        return 0
    match = re.search(r"adapter-(\d+)\.safetensors$", adapter_path.name)
    start_step = int(match.group(1)) if match else 0
    if reset_optimizer:
        print(f"starting with fresh optimizer state from adapter step {start_step}")
        return start_step
    optimizer_path = adapter_path.with_name(
        adapter_path.name.replace(".safetensors", ".optimizer.safetensors")
    )
    if not optimizer_path.exists():
        print(f"warning: optimizer checkpoint not found: {optimizer_path}")
        return start_step
    optimizer.init(model.trainable_parameters())
    state = mx.load(str(optimizer_path))
    optimizer.state = tree_unflatten(list(state.items()))
    mx.eval(optimizer.state)
    print(f"resumed optimizer state from step {start_step}")
    return start_step


def _evaluate(
    model: models.Lm,
    dataset: TokenDataset,
    batches: int,
    gradient_checkpointing: bool,
) -> float:
    model.eval()
    dataset.reset()
    losses = []
    for _ in range(batches):
        loss = training_loss(
            model,
            dataset.sample(),
            gradient_checkpointing=gradient_checkpointing,
        )
        mx.eval(loss)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def _fuse_lora(model: models.Lm) -> None:
    for layer in model.transformer.layers:
        for parent, name in (
            (layer.self_attn, "in_proj"),
            (layer.self_attn, "out_proj"),
            (layer.gating, "linear_in"),
            (layer.gating, "linear_out"),
        ):
            value = getattr(parent, name)
            if isinstance(value, LoRALinear):
                setattr(parent, name, value.fuse())
    if isinstance(model.text_emb, LoRAEmbedding):
        model.text_emb = model.text_emb.fuse()
    if isinstance(model.text_linear, LoRALinear):
        model.text_linear = model.text_linear.fuse()
    for slice_ in model.depformer.slices:
        for name in ("linear_in", "linear_out"):
            value = getattr(slice_, name)
            if isinstance(value, LoRALinear):
                setattr(slice_, name, value.fuse())
    if isinstance(model.depformer.slices[0].emb, LoRAEmbedding):
        model.depformer.slices[0].emb = model.depformer.slices[0].emb.fuse()


def _export_merged_checkpoint(
    model: models.Lm,
    output_dir: Path,
    config: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    _fuse_lora(model)
    merged = output_dir / "model.moshi_mlx.safetensors"
    model.save_weights(str(merged))
    shutil.copy2(config, output_dir / "config.json")
    print(f"wrote merged MLX checkpoint to {merged}")
    return merged


def _physical_memory() -> int:
    try:
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    except (OSError, subprocess.SubprocessError, ValueError):
        return 32 * GB


def _load_model(args: argparse.Namespace) -> models.Lm:
    config_data = json.loads(args.config.read_text())
    config = models.LmConfig.from_config_dict(config_data)
    model = models.Lm(config)
    model.set_dtype(mx.bfloat16)
    model.load_weights(str(args.weights), strict=True)
    inject_lora(
        model,
        args.rank,
        args.alpha,
        args.train_layers,
        adapt_input_embedding=args.adapt_input_embedding,
    )
    if args.resume is not None:
        model.load_weights(str(args.resume), strict=False)
    mx.eval(model.parameters())
    return model


def train(args: argparse.Namespace) -> None:
    memory_limit = int(args.memory_limit_gb * GB)
    mx.set_memory_limit(memory_limit)
    mx.set_cache_limit(int(args.cache_limit_gb * GB))
    mx.random.seed(args.seed)
    model = _load_model(args)
    if args.export_only:
        _export_merged_checkpoint(model, args.output_dir, args.config)
        return
    model.train()
    trainable = _trainable_count(model)
    print(f"trainable parameters: {trainable:,}")

    dataset = TokenDataset(
        args.data,
        args.sequence_frames,
        args.seed,
        args.min_text_tokens,
    )
    eval_dataset = (
        TokenDataset(
            args.eval_data,
            args.sequence_frames,
            args.seed + 1,
            args.min_text_tokens,
        )
        if args.eval_data is not None
        else None
    )
    optimizer = optim.AdamW(
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    start_step = _resume_optimizer(
        optimizer,
        model,
        args.resume,
        args.reset_optimizer,
    )
    loss_and_grad = nn.value_and_grad(
        model,
        lambda codes: training_loss(
            model,
            codes,
            gradient_checkpointing=not args.no_gradient_checkpointing,
        ),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_config.json").write_text(
        json.dumps(vars(args), default=str, indent=2, sort_keys=True) + "\n"
    )
    metrics_path = args.output_dir / "metrics.jsonl"
    if start_step == 0 and metrics_path.exists():
        metrics_path.unlink()
    if start_step >= args.steps:
        raise ValueError(
            f"checkpoint is already at step {start_step}; --steps must be larger"
        )
    for step in range(start_step + 1, args.steps + 1):
        started = time.perf_counter()
        accumulated = None
        losses = []
        mx.reset_peak_memory()
        for _ in range(args.grad_accum):
            loss, gradients = loss_and_grad(dataset.sample())
            mx.eval(loss, gradients)
            losses.append(float(loss.item()))
            accumulated = (
                gradients
                if accumulated is None
                else tree_map(lambda left, right: left + right, accumulated, gradients)
            )
        assert accumulated is not None
        gradients = tree_map(lambda value: value / args.grad_accum, accumulated)
        gradients = _clip_gradients(gradients, args.max_grad_norm)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state)
        elapsed = time.perf_counter() - started
        peak = mx.get_peak_memory() / GB
        print(
            f"step={step} loss={sum(losses) / len(losses):.4f} "
            f"seconds={elapsed:.2f} peak_gb={peak:.2f}"
        )
        metric = {
            "step": step,
            "train_loss": round(sum(losses) / len(losses), 6),
            "seconds": round(elapsed, 4),
            "peak_gb": round(peak, 4),
        }
        if step % args.save_every == 0 or step == args.steps:
            if eval_dataset is not None:
                eval_loss = _evaluate(
                    model,
                    eval_dataset,
                    args.eval_batches,
                    gradient_checkpointing=not args.no_gradient_checkpointing,
                )
                print(f"eval_step={step} loss={eval_loss:.4f}")
                metric["eval_loss"] = round(eval_loss, 6)
            _save_checkpoint(
                model,
                optimizer,
                args.output_dir,
                step,
            )
            mx.clear_cache()
        with metrics_path.open("a", encoding="utf-8") as handle:
            json.dump(metric, handle)
            handle.write("\n")

    if args.export_merged:
        _export_merged_checkpoint(model, args.output_dir, args.config)


def _tiny_config() -> LmConfig:
    main = TransformerConfig(
        d_model=32,
        num_heads=4,
        num_layers=2,
        dim_feedforward=128,
        causal=True,
        norm_first=True,
        bias_ff=False,
        bias_attn=False,
        layer_scale=None,
        context=64,
        max_period=10_000,
        use_conv_block=False,
        use_conv_bias=True,
        cross_attention=False,
        gating=True,
        norm="rms_norm",
        positional_embedding="rope",
        conv_layout=False,
        conv_kernel_size=3,
        kv_repeat=1,
        max_seq_len=64,
    )
    depth = TransformerConfig(
        d_model=16,
        num_heads=4,
        num_layers=2,
        dim_feedforward=64,
        causal=True,
        norm_first=True,
        bias_ff=False,
        bias_attn=False,
        layer_scale=None,
        context=4,
        max_period=10_000,
        use_conv_block=False,
        use_conv_bias=True,
        cross_attention=False,
        gating=True,
        norm="rms_norm",
        positional_embedding="none",
        conv_layout=False,
        conv_kernel_size=3,
        kv_repeat=1,
        max_seq_len=8,
    )
    return LmConfig(
        transformer=main,
        depformer=DepFormerConfig(transformer=depth, num_slices=2),
        text_in_vocab_size=65,
        text_out_vocab_size=64,
        audio_vocab_size=17,
        audio_codebooks=4,
        audio_delays=[0, 1, 0, 1],
        conditioners={},
    )


def self_test() -> None:
    model = models.Lm(_tiny_config())
    inject_lora(
        model,
        rank=4,
        alpha=8,
        train_layers=2,
        adapt_input_embedding=True,
    )
    codes = mx.concatenate(
        [
            mx.random.randint(4, 63, (1, 1, 12)),
            mx.random.randint(0, 16, (1, 4, 12)),
        ],
        axis=1,
    )
    loss_and_grad = nn.value_and_grad(model, lambda values: training_loss(model, values))
    optimizer = optim.AdamW(learning_rate=1e-3)
    loss, gradients = loss_and_grad(codes)
    optimizer.update(model, gradients)
    mx.eval(loss, model.parameters(), optimizer.state)
    if not math.isfinite(float(loss.item())):
        raise RuntimeError(f"non-finite self-test loss: {loss.item()}")
    trainable = _trainable_count(model)
    with tempfile.TemporaryDirectory(prefix="moshi-mlx-train-self-test-") as directory:
        root = Path(directory)
        config = root / "source-config.json"
        config.write_text("{}\n", encoding="utf-8")
        checkpoint = _export_merged_checkpoint(model, root / "output", config)
        reloaded = models.Lm(_tiny_config())
        reloaded.load_weights(str(checkpoint), strict=True)
        mx.eval(reloaded.parameters())
    print(
        f"MLX training self-test passed: loss={float(loss.item()):.4f}, "
        f"trainable={trainable:,}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--eval-data", type=Path)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--weights", type=Path, default=Path("model-mlx/model.moshi_mlx.safetensors"))
    parser.add_argument("--config", type=Path, default=Path("model-mlx/config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("training/uk_UA/mlx-runs/personaplex-uk"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--sequence-frames", type=int, default=128)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--min-text-tokens", type=int, default=4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--train-layers", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=4e-6)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--memory-limit-gb", type=float)
    parser.add_argument("--cache-limit-gb", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--export-merged", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--adapt-input-embedding", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return args
    if args.export_only and args.resume is None:
        parser.error("--export-only requires --resume")
    if args.data is None and not args.export_only:
        parser.error("--data is required unless --export-only is used")
    required_paths = [args.weights, args.config]
    if args.data is not None:
        required_paths.append(args.data)
    for path in required_paths:
        if not path.exists():
            parser.error(f"missing file: {path}")
    if args.resume is not None and not args.resume.exists():
        parser.error(f"missing adapter: {args.resume}")
    if args.eval_data is not None and not args.eval_data.exists():
        parser.error(f"missing eval manifest: {args.eval_data}")
    if args.eval_batches < 1:
        parser.error("--eval-batches must be positive")
    for name in ("steps", "sequence_frames", "grad_accum", "rank", "train_layers", "save_every"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.memory_limit_gb is None:
        args.memory_limit_gb = min(96.0, _physical_memory() / GB * 0.75)
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
    else:
        train(args)


if __name__ == "__main__":
    main()
