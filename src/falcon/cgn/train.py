"""CGN v2 training loop — DDP + fp16 + gradient accumulation + wandb."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast

from .config import get_config
from .model import CGNv2
from .tokenizer import CGNv2Tokenizer, UNK
from .data import build_dataloaders
from .generate import synthesize, tokens_to_audio, extract_plan_tokens

def _load_val_texts(val_jsonl: str) -> list[str]:
    """Load text fields from val JSONL for eval sampling."""
    texts = []
    with open(val_jsonl) as f:
        for line in f:
            data = json.loads(line)
            if data.get("text"):
                texts.append(data["text"])
    return texts


def train(args):
    # ── DDP setup ───────────────────────────────────────────────────────
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = f"cuda:{rank}"
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = args.device

    is_main = rank == 0

    # ── Tokenizer + config ──────────────────────────────────────────────
    tokenizer = CGNv2Tokenizer.load(args.tokenizer_json)
    config = get_config(
        args.model_size,
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
        cfg_drop_prob=args.cfg_drop_prob,
    )

    if is_main:
        print(f"Config: {args.model_size}, vocab={config.vocab_size}, "
              f"d_model={config.d_model}, layers={config.n_layers}")

    # ── Model ───────────────────────────────────────────────────────────
    model = CGNv2(config).to(device)
    if is_main:
        n_params = model.count_parameters()
        print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    if args.gradient_checkpointing:
        model.set_gradient_checkpointing(True)
        if is_main:
            print("Gradient checkpointing enabled")

    if ddp:
        model = DDP(model, device_ids=[rank])
    raw_model = model.module if ddp else model

    # ── Optimizer + scheduler ───────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    scaler = GradScaler(enabled=args.fp16)

    # ── Data ────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        args.train_jsonl, args.val_jsonl,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        rank=rank, world_size=world_size,
    )

    # Load val texts for eval audio sampling
    val_texts = _load_val_texts(args.val_jsonl) if is_main else []

    if is_main:
        print(f"Train: {len(train_loader.dataset)} samples, "
              f"Val: {len(val_loader.dataset)} samples, "
              f"Val texts for eval: {len(val_texts)}")

    steps_per_epoch = len(train_loader) // args.grad_accum_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = args.warmup_steps

    if is_main:
        print(f"Steps/epoch: {steps_per_epoch}, Total: {total_steps}, Warmup: {warmup_steps}")

    # ── wandb ───────────────────────────────────────────────────────────
    if is_main and args.wandb_project:
        import wandb
        wandb.init(
            project=args.wandb_project,
            config={
                "model_size": args.model_size,
                "vocab_size": config.vocab_size,
                "d_model": config.d_model,
                "n_layers": config.n_layers,
                "n_params": n_params,
                "batch_size": args.batch_size * world_size,
                "effective_batch": args.batch_size * world_size * args.grad_accum_steps,
                "lr": args.lr,
                "epochs": args.epochs,
                "fp16": args.fp16,
                "cfg_drop_prob": args.cfg_drop_prob,
                "codec": getattr(args, "codec", "snac"),
                "max_seq_len": args.max_seq_len,
            },
        )

    # ── Checkpoint resume ───────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    patience_counter = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latest_ckpt = output_dir / "latest.pt"
    if latest_ckpt.exists() and not args.fresh:
        if is_main:
            print(f"Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        patience_counter = ckpt.get("patience_counter", 0)

    # ── Training loop ───────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if ddp and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        epoch_tokens = 0
        micro_step = 0
        t0 = time.time()

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # ── CFG: randomly mask linguistic prefix ────────────────
            if config.cfg_drop_prob > 0 and torch.rand(1).item() < config.cfg_drop_prob:
                # Replace non-special, non-audio tokens with UNK
                from .tokenizer import NUM_SPECIAL, PLAN_START, AUDIO_START
                mask = (input_ids >= NUM_SPECIAL) & (input_ids < tokenizer.audio_offset)
                # Keep PLAN_START and AUDIO_START
                mask = mask & (input_ids != PLAN_START) & (input_ids != AUDIO_START)
                input_ids = input_ids.clone()
                input_ids[mask] = UNK

            # ── Forward ────────────────────────────────────────────
            with autocast(enabled=args.fp16):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, config.vocab_size),
                    labels.view(-1),
                    ignore_index=-100,
                )
                loss = loss / args.grad_accum_steps

            scaler.scale(loss).backward()

            # Count tokens for logging
            n_tokens = (labels != -100).sum().item()
            epoch_loss += loss.item() * args.grad_accum_steps * n_tokens
            epoch_tokens += n_tokens

            micro_step += 1
            if micro_step % args.grad_accum_steps == 0:
                # ── Gradient step ──────────────────────────────────
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                ).item()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                # ── LR schedule (cosine with warmup) ───────────────
                global_step += 1
                if global_step <= warmup_steps:
                    lr = args.lr * global_step / warmup_steps
                else:
                    progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
                    lr = args.lr * 0.5 * (1 + math.cos(math.pi * progress))
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                # ── Logging (every step to wandb, print at log_interval) ─
                if is_main:
                    step_loss = loss.item() * args.grad_accum_steps
                    avg_loss = epoch_loss / max(epoch_tokens, 1)
                    elapsed = time.time() - t0
                    tok_per_sec = epoch_tokens / elapsed if elapsed > 0 else 0
                    gpu_mem = torch.cuda.max_memory_allocated(device) / 1e9

                    # Log every step to wandb
                    if args.wandb_project:
                        import wandb
                        wandb.log({
                            "train/loss": avg_loss,
                            "train/step_loss": step_loss,
                            "train/lr": lr,
                            "train/tokens_per_sec": tok_per_sec,
                            "train/gpu_mem_gb": gpu_mem,
                            "train/grad_norm": grad_norm,
                            "train/epoch": epoch + 1,
                        }, step=global_step)

                    # Print at log_interval
                    if global_step % args.log_interval == 0:
                        print(f"[E{epoch+1} S{global_step}] "
                              f"loss={avg_loss:.4f} lr={lr:.2e} "
                              f"tok/s={tok_per_sec:.0f} mem={gpu_mem:.1f}GB "
                              f"gnorm={grad_norm:.2f}")

                # ── Checkpoint ─────────────────────────────────────
                if is_main and global_step % args.save_interval == 0:
                    _save_checkpoint(raw_model, optimizer, scaler, epoch, global_step, output_dir)

                # ── Mid-training eval audio ───────────────────────
                if is_main and args.wandb_project and val_texts \
                        and global_step % args.eval_interval == 0:
                    _eval_audio(raw_model, tokenizer, device, global_step, args, val_texts)

        # ── End of epoch ────────────────────────────────────────────────
        avg_epoch_loss = epoch_loss / max(epoch_tokens, 1)

        # Validation
        val_loss = _validate(model, val_loader, config, device, args.fp16)

        if is_main:
            print(f"[Epoch {epoch+1}/{args.epochs}] "
                  f"train_loss={avg_epoch_loss:.4f} val_loss={val_loss:.4f}")

            if args.wandb_project:
                import wandb
                wandb.log({
                    "val/loss": val_loss,
                    "train/avg_loss": avg_epoch_loss,
                }, step=global_step)

            _save_checkpoint(raw_model, optimizer, scaler, epoch + 1, global_step,
                             output_dir, best_val_loss=best_val_loss,
                             patience_counter=patience_counter)

            # Best checkpoint tracking
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_path = output_dir / "best.pt"
                torch.save({
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                }, best_path)
                print(f"  New best val_loss={val_loss:.4f}, saved {best_path}")
            else:
                patience_counter += 1
                print(f"  Val loss did not improve (best={best_val_loss:.4f}, "
                      f"patience={patience_counter}/{args.patience})")

            # Audio eval — generate sample from val set and log to wandb
            if args.wandb_project and val_texts:
                _eval_audio(raw_model, tokenizer, device, global_step, args, val_texts)

            # Early stopping
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"Early stopping: val loss did not improve for {args.patience} epochs")
                break

    if is_main:
        print(f"Training complete. Best val_loss={best_val_loss:.4f}")
        if args.wandb_project:
            import wandb
            wandb.finish()

    if ddp:
        dist.destroy_process_group()


@torch.no_grad()
def _validate(model, val_loader, config, device, fp16):
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        with autocast(enabled=fp16):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )

        n_tokens = (labels != -100).sum().item()
        total_loss += loss.item()
        total_tokens += n_tokens

    model.train()
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def _eval_audio(model, tokenizer, device, global_step, args, val_texts):
    """Generate 1 random eval audio sample from val set and log to wandb."""
    import wandb
    from .tokenizer import AUDIO_START as _AUDIO_START, PLAN_START as _PLAN_START

    codec_type = getattr(args, "codec", "snac")

    model.eval()
    try:
        if codec_type == "qwen12hz":
            from .qwen_codec import QwenCodec
            codec = QwenCodec(device=device)
        else:
            from .snac_codec import SNACCodec
            codec = SNACCodec(device=device)
    except (ImportError, Exception) as e:
        print(f"  [eval_audio] skipped — codec unavailable: {e}")
        model.train()
        return

    # Randomly pick 1 utterance from val set
    text = random.choice(val_texts)
    audio_logs = {}
    metric_logs = {}

    try:
        _, token_ids = synthesize(
            model, tokenizer, text,
            device=device,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            max_new_tokens=1024,
            cfg_scale=1.0,  # No CFG during early training
            flite_bin=args.flite_bin,
        )

        # Diagnostics
        plan = extract_plan_tokens(token_ids, tokenizer)
        n_audio = sum(1 for t in token_ids if tokenizer.is_audio_token(t))

        if codec_type == "qwen12hz":
            # Qwen: count tokens and check level distribution
            n_levels = tokenizer.snac_n_levels
            caption = (f"{text[:80]} | plan={len(plan)} audio={n_audio} levels={n_levels}")
            print(f"  Eval: {text[:60]}... "
                  f"(plan={len(plan)} audio={n_audio} levels={n_levels})")
        else:
            # SNAC: check level distribution
            level_counts = {0: 0, 1: 0, 2: 0}
            for t in token_ids:
                if tokenizer.is_audio_token(t):
                    lev, _ = tokenizer.decode_snac(t)
                    level_counts[lev] = level_counts.get(lev, 0) + 1

            ratio_str = f"c={level_counts[0]} m={level_counts[1]} f={level_counts[2]}"
            expected_ratio = level_counts[0] > 0 and \
                abs(level_counts[1] / max(level_counts[0], 1) - 2.0) < 0.5 and \
                abs(level_counts[2] / max(level_counts[0], 1) - 4.0) < 1.0

            caption = (f"{text[:80]} | plan={len(plan)} audio={n_audio} "
                       f"{ratio_str} valid_ratio={expected_ratio}")
            metric_logs["eval/valid_ratio"] = 1.0 if expected_ratio else 0.0
            print(f"  Eval: {text[:60]}... "
                  f"(plan={len(plan)} audio={n_audio} {ratio_str} "
                  f"ratio_ok={expected_ratio})")

        waveform = tokens_to_audio(
            token_ids, tokenizer, codec, codec_type=codec_type,
        )
        audio_np = waveform.squeeze().cpu().numpy()

        audio_logs["eval/audio"] = wandb.Audio(
            audio_np, sample_rate=24000, caption=caption,
        )
        metric_logs["eval/n_audio_tokens"] = n_audio
        metric_logs["eval/n_plan_tokens"] = len(plan)

    except Exception as e:
        print(f"  Eval failed: {e}")

    if audio_logs:
        wandb.log({**audio_logs, **metric_logs}, step=global_step)

    model.train()
    del codec


def _save_checkpoint(model, optimizer, scaler, epoch, global_step, output_dir,
                     best_val_loss=float("inf"), patience_counter=0):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
    }
    path = output_dir / f"checkpoint_{global_step}.pt"
    torch.save(ckpt, path)
    # Also save as latest for easy resume
    latest = output_dir / "latest.pt"
    torch.save(ckpt, latest)
    print(f"  Saved checkpoint: {path}")

    # Keep only last 3 numbered checkpoints to avoid filling disk
    numbered = sorted(output_dir.glob("checkpoint_*.pt"))
    for old in numbered[:-3]:
        old.unlink()
        print(f"  Deleted old checkpoint: {old.name}")


def main():
    parser = argparse.ArgumentParser(description="CGN v2 training")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--val_jsonl", type=str, required=True)
    parser.add_argument("--tokenizer_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints/cgn_v2")
    parser.add_argument("--model_size", type=str, default="small", choices=["tiny", "mini", "small", "base"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", dest="fp16", action="store_false")
    parser.add_argument("--cfg_drop_prob", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--wandb_project", type=str, default="festvox3.0")
    parser.add_argument("--eval_interval", type=int, default=1000,
                        help="Generate eval audio every N steps (in addition to end of epoch)")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                        help="Enable gradient checkpointing to save memory")
    parser.add_argument("--flite_bin", type=str, default="flite")
    parser.add_argument("--patience", type=int, default=0,
                        help="Early stopping patience (0 = disabled)")
    parser.add_argument("--fresh", action="store_true", default=False,
                        help="Start fresh, ignore existing checkpoints in output_dir")
    parser.add_argument("--codec", type=str, default="snac",
                        choices=["snac", "qwen12hz"],
                        help="Audio codec (affects eval audio decode only; data is pre-tokenized)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
