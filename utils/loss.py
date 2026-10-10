import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# GT-supervised component losses  (called per output via LossFunc)
# ---------------------------------------------------------------------------

def focal_loss(pred, mask, gamma=2.0, alpha=0.75):
    """
    Focal loss (sigmoid variant).

    Why instead of plain BCE:
      Polyp regions occupy a tiny fraction of colonoscopy frames (often <5%).
      Plain BCE treats every pixel equally, so the model is overwhelmed by
      easy background pixels and learns to under-predict foreground.
      Focal loss down-weights well-classified easy pixels via (1-p_t)^gamma,
      forcing training to focus on *hard* (small-polyp) pixels.

      alpha=0.75 additionally up-weights the positive (polyp) class to
      compensate for the class-frequency imbalance.
    """
    eps  = 1e-6
    pred = pred.clamp(eps, 1.0 - eps)
    pos  = -alpha       * (1 - pred) ** gamma * mask       * torch.log(pred)
    neg  = -(1 - alpha) * pred       ** gamma * (1 - mask) * torch.log(1 - pred)
    return (pos + neg).mean()


def dice_loss(pred, mask, smooth=1.0):
    """
    Soft Dice loss — directly optimises the mDice evaluation metric.
    Better than IoU for class-imbalanced binary masks.
    """
    pf    = pred.view(pred.shape[0], -1)
    mf    = mask.view(mask.shape[0], -1)
    inter = (pf * mf).sum(dim=1)
    dice  = (2.0 * inter + smooth) / (pf.sum(dim=1) + mf.sum(dim=1) + smooth)
    return (1.0 - dice).mean()


def boundary_loss(pred, mask, kernel_size=5):
    """
    Boundary-aware weighted BCE.
    Up-weights boundary pixels ×5 to train BARM's refinement path.
    Boundary ring extracted via morphological dilation − erosion on GT.
    """
    pad      = kernel_size // 2
    dilated  =  F.max_pool2d( mask, kernel_size=kernel_size, stride=1, padding=pad)
    eroded   = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)
    boundary = (dilated - eroded).clamp(0, 1)

    eps    = 1e-6
    pred   = pred.clamp(eps, 1.0 - eps)
    bce    = F.binary_cross_entropy(pred, mask, reduction='none')
    weight = 1.0 + 4.0 * boundary
    return (bce * weight).mean()


def ssim_loss(pred, mask, window_size=7):
    """
    SSIM structural loss — rewards correct polyp shape, not just pixel count.
    Uses avg_pool2d as a fast box-filter approximation to Gaussian SSIM.
    """
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    pad     = window_size // 2

    mu_p   = F.avg_pool2d(pred,        window_size, stride=1, padding=pad)
    mu_m   = F.avg_pool2d(mask,        window_size, stride=1, padding=pad)
    mu_p2  = mu_p * mu_p
    mu_m2  = mu_m * mu_m
    mu_pm  = mu_p * mu_m

    sig_p2 = F.avg_pool2d(pred * pred, window_size, stride=1, padding=pad) - mu_p2
    sig_m2 = F.avg_pool2d(mask * mask, window_size, stride=1, padding=pad) - mu_m2
    sig_pm = F.avg_pool2d(pred * mask,  window_size, stride=1, padding=pad) - mu_pm

    ssim_map = ((2 * mu_pm + C1) * (2 * sig_pm + C2)) / \
               ((mu_p2 + mu_m2 + C1) * (sig_p2 + sig_m2 + C2))
    return (1.0 - ssim_map).mean()


def LossFunc(pred, mask):
    """
    Per-output GT-supervised loss: Focal + Dice + Boundary + SSIM.
    Called independently for fine mask, coarse mask, and each side logit.
    """
    mask = mask.float()
    return (focal_loss(pred, mask, gamma=2.0, alpha=0.75)
            + dice_loss(pred, mask)
            + 0.5 * boundary_loss(pred, mask, kernel_size=5)
            + 0.3 * ssim_loss(pred, mask))


# ---------------------------------------------------------------------------
# PSL-inspired regularization losses  (called once per step on final output)
# ---------------------------------------------------------------------------

def _get_boundary_mask(mask, kernel_size=5):
    """Extract GT boundary ring via dilation − erosion (shared utility)."""
    pad      = kernel_size // 2
    dilated  =  F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    eroded   = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)
    return (dilated - eroded).clamp(0, 1)


def appearance_weighted_tv_loss(pred, img, edge_strength=5.0):
    """
    PSL Rule: neighbor(x,y) ∧ similar_appearance(x,y) → same_label(x,y)

    Smoothness between two neighbouring pixels is weighted by how similar
    their colour is.  At a real edge (large image gradient), the weight
    approaches 0 — the model is free to predict different labels.
    In a homogeneous region (small image gradient), the weight is ~1 —
    neighbouring pixels should have the same prediction.

    This directly addresses fragmented predictions on small polyps
    (ETIS-Lari, CVC-ColonDB) where holes and isolated false pixels appear
    because no spatial coherence is enforced.

    img  : (B, 3, H, W) normalised image — used only for gradient direction
    pred : (B, 1, H, W) sigmoid probabilities
    edge_strength = 5.0 chosen for ImageNet-normalised images where
      typical edge gradients are 0.1–0.5; exp(-5×0.3) ≈ 0.22 at edges,
      exp(-5×0.05) ≈ 0.78 in flat regions.
    """
    # Mean absolute colour difference between vertical / horizontal neighbours
    img_dy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]).mean(dim=1, keepdim=True)  # (B,1,H-1,W)
    img_dx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]).mean(dim=1, keepdim=True)  # (B,1,H,W-1)

    # Smoothness weights: large in flat areas, small at colour edges
    w_y = torch.exp(-edge_strength * img_dy)
    w_x = torch.exp(-edge_strength * img_dx)

    # Prediction difference between neighbours
    pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])

    return (w_y * pred_dy).mean() + (w_x * pred_dx).mean()


def consistency_loss(fine, coarse, side_list):
    """
    PSL Rule: coarse_polyp(x) → fine_polyp(x)
              side_i_polyp(x) → fine_polyp(x)

    Soft implication A → B is violated when A is high but B is low.
    PSL encodes this as the penalty: max(0, A − B).

    This rule enforces that the final fine prediction does not contradict
    the coarser intermediate predictions.  The coarse decoder and side heads
    aggregate multi-scale ViT features independently; if they agree a region
    is polyp, the fine output should not predict background there.

    .detach() on coarse/side: they act as soft constraints (teachers),
    not as learnable paths — avoids circular gradient loops.

    All inputs must be sigmoid probabilities at the same spatial size.
    """
    # Penalise where coarse says polyp but fine doesn't
    loss = torch.mean(torch.clamp(coarse.detach() - fine, min=0.0))

    # HLIF side outputs act as detached consistency targets.
    for s in side_list:
        loss = loss + 0.1 * torch.mean(torch.clamp(s.detach() - fine, min=0.0))

    return loss


def area_plausibility_loss(pred, min_ratio=0.001, max_ratio=0.5):
    """
    PSL Rule: min_area ≤ ∑ polyp(x) ≤ max_area

    A valid polyp occupies between 0.1% and 50% of the frame.
    Penalises degenerate predictions (all-background or all-polyp) that
    can appear during early training when the CNN encoder is unstable.

    The penalty is exactly 0 whenever the area is in the plausible range —
    it only fires at extremes, so it never interferes with normal training.

    min_ratio=0.001 → prevents silent all-background collapse
    max_ratio=0.5   → prevents all-foreground collapse
    """
    area      = pred.mean(dim=(2, 3))                           # (B, 1)
    too_small = torch.clamp(min_ratio - area, min=0.0)
    too_large = torch.clamp(area - max_ratio, min=0.0)
    return (too_small + too_large).mean()


def boundary_sharpness_loss(pred, mask, kernel_size=5):
    """
    PSL Rule: boundary(x) → certain_label(x)   [¬uncertain(x) on boundaries]

    Pixels near the polyp edge should commit to polyp or background —
    not hover near 0.5.  Uncertainty = 1 − |2p − 1| peaks at p=0.5
    and is 0 at p=0 or p=1.  We penalise this uncertainty specifically
    on GT boundary pixels, pushing the model to make sharp transitions.

    Complements the existing boundary_loss in LossFunc (which upweights BCE
    at boundaries) by additionally penalising indecision independently of
    the GT direction.  Together they drive the BARM refinement path to
    produce crisp polyp outlines.
    """
    boundary    = _get_boundary_mask(mask.float(), kernel_size)   # (B,1,H,W)
    uncertainty = 1.0 - torch.abs(2.0 * pred - 1.0)              # 0 at {0,1}, 1 at 0.5
    return (uncertainty * boundary).mean()


def psl_regularization(fine, coarse, side_list, mask, img):
    """
    Combined PSL regularization — called once per training step on the
    final fine prediction, using all available model outputs and the raw image.

    Rules and weights:
      0.10 × appearance_weighted_tv   — spatial coherence, fixes fragmentation
      0.30 × consistency              — multi-output agreement
      0.50 × area_plausibility        — prevents degenerate collapse
      0.20 × boundary_sharpness       — sharpens predicted outline

    Weights are intentionally small (total contribution ≈ 5–10% of LossFunc)
    so PSL acts as a regularizer that guides the solution space rather than
    dominating the GT-supervised signal.

    Args:
        fine      : (B,1,H,W) final mask probabilities  [sigmoid applied]
        coarse    : (B,1,H,W) coarse mask probabilities [sigmoid applied]
        side_list : list of (B,1,H,W) side-logit probs  [sigmoid applied]
        mask      : (B,1,H,W) binary GT mask             [float 0/1]
        img       : (B,3,H,W) normalised input image     [for TV edge weights]
    """
    mask = mask.float()

    l_tv      = appearance_weighted_tv_loss(fine, img)
    l_consist = consistency_loss(fine, coarse, side_list)
    l_area    = area_plausibility_loss(fine)
    l_sharp   = boundary_sharpness_loss(fine, mask)

    return (0.10 * l_tv
            + 0.30 * l_consist
            + 0.50 * l_area
            + 0.20 * l_sharp)
