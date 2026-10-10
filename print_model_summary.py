import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from model.bahnet import BAHNet

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

model = BAHNet(img_size=512).to(device)
model.eval()

# ── 1. torchsummary (layer-by-layer table) ──────────────────────────────────
try:
    from torchsummary import summary
    print("=" * 80)
    print("TORCHSUMMARY  (layer-by-layer)")
    print("=" * 80)
    summary(model, input_size=(3, 512, 512), device=device)
except Exception as e:
    print(f"[torchsummary] skipped ({e})\n")

# ── 2. PyTorch repr (full module tree) ─────────────────────────────────────
print("\n" + "=" * 80)
print("PYTORCH MODULE TREE")
print("=" * 80)
print(model)

# ── 3. Parameter counts per sub-module ─────────────────────────────────────
print("\n" + "=" * 80)
print("PARAMETER COUNTS PER SUB-MODULE")
print("=" * 80)
total = 0
trainable = 0
for name, module in model.named_children():
    params      = sum(p.numel() for p in module.parameters())
    t_params    = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total      += params
    trainable  += t_params
    print(f"  {name:<25} total={params:>12,}  trainable={t_params:>12,}")

print("-" * 80)
print(f"  {'TOTAL':<25} total={total:>12,}  trainable={trainable:>12,}")

# ── 4. Quick forward-pass shape check ──────────────────────────────────────
# Use a fresh instance so no stale torchsummary hooks interfere
print("\n" + "=" * 80)
print("FORWARD PASS  (input 1x3x512x512)")
print("=" * 80)
model_fp = BAHNet(img_size=512).to(device)
model_fp.eval()
with torch.no_grad():
    dummy = torch.randn(1, 3, 512, 512).to(device)
    mask, coarse_mask, side_logits = model_fp(dummy)

print(f"  mask         shape: {tuple(mask.shape)}")
print(f"  coarse_mask  shape: {tuple(coarse_mask.shape)}")
if isinstance(side_logits, (list, tuple)):
    for i, s in enumerate(side_logits):
        print(f"  side_logit[{i}] shape: {tuple(s.shape)}")
else:
    print(f"  side_logits  shape: {tuple(side_logits.shape)}")
