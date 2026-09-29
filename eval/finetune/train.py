"""Single-device (Apple MPS / CPU / CUDA) port of Laya's public RLCD fine-tuning loop.

Adapted from notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb (Apache-2.0, Convai Innovations):
the DDP / NCCL / GradScaler machinery is removed; loss (policy gradient + soft CE), optimiser groups,
cosine schedule and per-type temperature calibration are unchanged.

Usage: python train.py <base_checkpoint_dir> <items_dir> <out_dir> [--epochs 4] [--micro-batch 16] [--smoke N]
items_dir holds train_items.pt / dev_items.pt from build_items.py. The calibration slice is taken from dev.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya.common import QTYPES, build_model, proper_reward


def collate(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {"input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask, "target": target,
            "qtype": torch.tensor([it["qtype"] for it in items])}


def fit_one_temp(sel):
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, : len(z)] = torch.tensor(z)
        T[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


@torch.no_grad()
def evaluate(model, items, tok, device, amp):
    model.eval()
    correct, n, preds = 0, 0, []
    for i in range(0, len(items), 32):
        chunk = items[i:i + 32]
        b = collate(chunk, tok.pad_token_id)
        with amp():
            logits, _ = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["marker_pos"].to(device),
                              b["marker_mask"].to(device), b["qtype"].to(device))
        logits = logits.float().cpu().numpy()
        for r, it in enumerate(chunk):
            k = len(it["markers"])
            preds.append((it["qtype"], logits[r, :k], it["target"]))
            correct += int(np.argmax(logits[r, :k]) == it["label"])
            n += 1
    model.train()
    return correct / max(1, n), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("items"); ap.add_argument("out")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr-enc", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--smoke", type=int, default=0, help="train on only N items for a few steps to measure speed")
    ap.add_argument("--amp", default="fp32", choices=["bf16", "fp32"])  # fp32 was ~2x faster than bf16 on M4 Pro MPS
    a = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    cfg = json.load(open(os.path.join(a.base, "rl_agent_config.json")))
    cfg["gradient_checkpointing"] = False
    tok = AutoTokenizer.from_pretrained(os.path.join(a.base, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(a.base, "encoder"))
    model.load_state_dict(load_file(os.path.join(a.base, "model.safetensors")), strict=True)
    model.to(device).train()

    def amp():
        if a.amp == "fp32":
            return torch.autocast(device.type, enabled=False)
        return torch.autocast(device.type, dtype=torch.bfloat16)

    train_items = torch.load(os.path.join(a.items, "train_items.pt"), weights_only=False)
    dev_items = torch.load(os.path.join(a.items, "dev_items.pt"), weights_only=False)
    if a.smoke:
        train_items = train_items[: a.smoke]
    # calibration slice comes from dev (never trained on); the rest of dev is for reporting only
    rng = random.Random(20260929)
    dev_order = list(range(len(dev_items))); rng.shuffle(dev_order)
    calib = [dev_items[i] for i in dev_order[: len(dev_items) // 2]]
    dev_only = [dev_items[i] for i in dev_order[len(dev_items) // 2:]]

    enc = [p for n, p in model.named_parameters() if "encoder." in n]
    head = [p for n, p in model.named_parameters() if "encoder." not in n]
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr_enc}, {"params": head, "lr": a.lr_head}], weight_decay=0.01)
    steps_per_epoch = max(1, len(train_items) // (a.micro_batch * a.accum))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_per_epoch * a.epochs, eta_min=1e-6)

    print(f"device={device} amp={a.amp} train={len(train_items)} calib={len(calib)} dev={len(dev_only)} "
          f"epochs={a.epochs} eff_batch={a.micro_batch * a.accum}", flush=True)
    t0 = time.time()
    for epoch in range(a.epochs):
        random.Random(42 + epoch).shuffle(train_items)
        sigma = 0.4 + (0.1 - 0.4) * epoch / max(1, a.epochs - 1)
        tot, nb, accum = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for i in range(0, len(train_items), a.micro_batch):
            chunk = train_items[i:i + a.micro_batch]
            b = collate(chunk, tok.pad_token_id)
            with amp():
                logits, act = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["marker_pos"].to(device),
                                    b["marker_mask"].to(device), b["qtype"].to(device))
            logits = logits.float()
            mask = b["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = b["target"].to(device)
            eps = torch.randn((4,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), b["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / a.accum + 0.0 * act.float().sum()
            loss.backward()
            accum += 1
            if accum % a.accum == 0 or i + a.micro_batch >= len(train_items):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            tot += loss.item() * a.accum; nb += 1
            if nb % 25 == 0:
                print(f"  ep {epoch + 1}/{a.epochs} step {nb} loss {loss.item() * a.accum:.4f} reward {r.mean().item():.3f} "
                      f"{(time.time() - t0):.0f}s", flush=True)
            if a.smoke and nb >= 6:
                break
        acc, _ = evaluate(model, dev_only, tok, device, amp)
        print(f"=== epoch {epoch + 1} done {time.time() - t0:.0f}s avg_loss {tot / max(1, nb):.4f} dev_acc {acc:.3f}", flush=True)
        if a.smoke:
            return

    _, preds = evaluate(model, calib, tok, device, amp)
    temps = [1.2, 1.2, 1.2]
    for qt in range(3):
        sel = [(z, t) for q_type, z, t in preds if q_type == qt]
        if sel:
            temps[qt] = fit_one_temp(sel)
    print("temperatures (choice, score, noul):", [round(t, 3) for t in temps])
    os.makedirs(a.out, exist_ok=True)
    save_file({k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}, os.path.join(a.out, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(a.out, "encoder"))
    tok.save_pretrained(os.path.join(a.out, "tokenizer"))
    cfg["fine_tuned"] = True
    cfg["model_name"] = "laya-ja-turn"
    cfg["temperature"] = temps
    cfg.pop("temperature_by_options", None)
    json.dump(cfg, open(os.path.join(a.out, "rl_agent_config.json"), "w"), indent=2)
    print("saved to", a.out)


if __name__ == "__main__":
    main()
