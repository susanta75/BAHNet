import torch
from torch import nn
from .common import LayerNorm2d
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# CBAM attention for the CNN encoder
# ---------------------------------------------------------------------------

class ChannelAttention(nn.Module):
    """
    Channel attention via Squeeze-and-Excitation with dual pooling.

    Answers: which feature channels are most relevant for this image?

    Why dual pooling (avg + max):
      AvgPool captures the overall activation level of each channel —
      useful when a feature type is broadly present.
      MaxPool captures peak responses — useful when a feature type is
      strongly present in at least one location even if rare.
      Summing both before sigmoid gives the MLP richer statistics to
      work with than either alone.

    Generalisation benefit:
      Different colonoscope models produce different colour temperatures.
      A scope with cooler white balance activates different channels than
      Kvasir's warmer tones.  Channel attention lets the encoder re-weight
      which detectors are active per image rather than using a fixed
      channel combination — this is the primary mechanism for adapting
      to unseen scope types at test time.

    reduction=4 keeps the bottleneck at in_channels/4, balancing
    expressiveness against overfitting on 1000 training images.
    """
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        reduced = max(in_channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, in_channels, bias=False),
        )

    def forward(self, x):
        avg  = x.mean(dim=(2, 3))              # (B, C)  global average
        maxv = x.amax(dim=(2, 3))              # (B, C)  global max
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(maxv))  # (B, C)
        return x * attn.unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    """
    Spatial attention via channel-pooled convolution.

    Answers: which spatial locations should the encoder focus on?

    Why this matters for colonoscopy generalisation:
      Every colonoscope produces a dark circular vignette border — the
      black ring around the circular field of view.  The exact shape and
      size of this ring differs across scope models.  Without spatial
      attention, the CNN must learn to ignore this ring from Kvasir alone;
      at test time a differently-shaped ring appears and misleads the model.
      Spatial attention can learn "suppress the dark border, amplify the
      central tissue region" in a general way that transfers across scopes.

      Additionally, polyps are almost always in the center 70% of the frame
      (the endoscopist aims the scope at suspicious regions).  Spatial
      attention reinforces this prior without hard-coding it.

    kernel_size=7 gives a large enough receptive field to capture the
    broad border-vs-center distinction without being too global.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg  = x.mean(dim=1, keepdim=True)    # (B, 1, H, W)
        maxv = x.amax(dim=1, keepdim=True)    # (B, 1, H, W)
        attn = torch.sigmoid(self.conv(torch.cat([avg, maxv], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Applies channel attention first then spatial attention — this order
    (established in the original CBAM paper) outperforms the reverse because
    channel attention globally re-calibrates which features matter before
    spatial attention decides where they matter.

    Parameters added per stage:
      CBAM(32,  reduction=4): ~  600 params
      CBAM(64,  reduction=4): ~ 2150 params
      CBAM(128, reduction=4): ~ 8300 params
      Total addition:         ~11 050 params  (negligible)
    """
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size=7)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class MSEB(nn.Module):
    """Pool multi-scale feature statistics and restore spatial detail."""
    def __init__(self, channels, bins=(3, 6, 9, 12)):
        super().__init__()
        self.bins = bins
        self.stat_proj = nn.ModuleList([
            nn.Conv2d(4, channels, 1) for _ in bins
        ])
        self.pointwise = nn.ModuleList([
            nn.Conv2d(channels, channels, 1) for _ in bins
        ])
        self.depthwise = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
            for _ in bins
        ])
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.fuse = nn.Conv2d(channels * (len(bins) + 1), channels, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        branches = [self.local(x)]
        for size, stats_proj, pointwise, depthwise in zip(
            self.bins, self.stat_proj, self.pointwise, self.depthwise
        ):
            pooled = F.adaptive_avg_pool2d(x, (size, size))
            mean = pooled.mean(dim=1, keepdim=True)
            var = pooled.var(dim=1, keepdim=True, unbiased=False)
            maximum = pooled.amax(dim=1, keepdim=True)
            std = var.clamp_min(1e-6).sqrt()
            stats = torch.cat((mean, var, maximum, std), dim=1)
            refined = F.relu(pointwise(pooled) + stats_proj(stats), inplace=True)
            refined = F.relu(depthwise(refined), inplace=True)
            branches.append(F.interpolate(refined, (h, w), mode="bilinear", align_corners=False))
        return self.fuse(torch.cat(branches, dim=1))


class DAB(nn.Module):
    """Low-rank domain adaptation with four pooled spatial scales."""
    def __init__(self, embed_dim=768, reduced_dim=256, bins=(3, 6, 9, 12)):
        super().__init__()
        self.down = nn.Linear(embed_dim, reduced_dim)
        self.mseb = MSEB(reduced_dim, bins)
        self.local = nn.Conv2d(reduced_dim, reduced_dim, 3, padding=1,
                               groups=reduced_dim)
        self.fuse = nn.Conv2d(reduced_dim * 2, reduced_dim, 1)
        self.up = nn.Linear(reduced_dim, embed_dim)

    def forward(self, tokens):
        reduced = F.relu(self.down(tokens), inplace=True).permute(0, 3, 1, 2).contiguous()
        multiscale = self.mseb(reduced)
        local = self.local(reduced)
        update = self.fuse(torch.cat((multiscale, local), dim=1))
        update = update.permute(0, 2, 3, 1).contiguous()
        return tokens + self.up(update)


# ---------------------------------------------------------------------------
# Lightweight parallel CNN encoder  (with CBAM after every stage)
# ---------------------------------------------------------------------------

class SPPE(nn.Module):
    """
    Parallel 3-stage CNN that runs alongside the ViT encoder to recover
    fine-grained spatial detail that the ViT structurally loses.

    Why needed:
      The ViT backbone tokenises with a 16×16 patch stride so the finest
      grid it operates on is 32×32 (for 512×512 input).  Sub-patch structure
      — thin boundaries, small polyps — is averaged away before block 0.
      This CNN preserves 256×256, 128×128 and 64×64 detail that feeds
      BARM via an FPN.

    CBAM at every stage:
      After each downsampling block a CBAM module re-calibrates the features:
      • Channel attention: adapts to the current image's colour/scope profile
      • Spatial attention: suppresses vignette borders, amplifies tissue center
      This makes each scale's output more discriminative before it is passed
      to the FPN, and is especially beneficial for cross-dataset generalisation
      where scope appearance varies significantly.

    Output channels (fixed, matched to BARM FPN):
      s2 : (B,  32, H/2, W/2)
      s4 : (B,  64, H/4, W/4)
      s8 : (B, 128, H/8, W/8)
    """
    def __init__(self, norm, act):
        super().__init__()
        # stride-2: 512→256,  out-channels 32
        self.stage1 = nn.Sequential(
            nn.Conv2d(3,  32, 3, stride=2, padding=1, bias=False),
            norm(32),
            act(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            norm(32),
            act(),
        )
        self.cbam1 = CBAM(32,  reduction=4)

        # stride-4: 256→128,  out-channels 64
        self.stage2 = nn.Sequential(
            nn.Conv2d(32,  64, 3, stride=2, padding=1, bias=False),
            norm(64),
            act(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            norm(64),
            act(),
        )
        self.cbam2 = CBAM(64,  reduction=4)

        # stride-8: 128→64,  out-channels 128
        self.stage3 = nn.Sequential(
            nn.Conv2d(64,  128, 3, stride=2, padding=1, bias=False),
            norm(128),
            act(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            norm(128),
            act(),
        )
        self.cbam3 = CBAM(128, reduction=4)

    def forward(self, x):
        s2 = self.cbam1(self.stage1(x))   # (B,  32, H/2, W/2)
        s4 = self.cbam2(self.stage2(s2))  # (B,  64, H/4, W/4)
        s8 = self.cbam3(self.stage3(s4))  # (B, 128, H/8, W/8)
        return s2, s4, s8


class EdgeEnhancer(nn.Module):
    """
    Dual-scale Difference-of-Gaussians edge enhancer.

    Why improved:
      - Original used a single 3×3 AvgPool diff, which only captures very thin
        edges at one scale.
      - Polyp boundaries vary in thickness; a 5×5 pool captures slightly thicker
        / coarser boundary signals that complement the fine 3×3 signal.
      - A lightweight channel-attention gate (SE-block style) learns to weight
        the fused edge response channel-by-channel, instead of treating all
        channels equally.
      - The original used Sigmoid as the conv activation inside EdgeEnhancer;
        this is replaced with the passed `act` (ReLU/GELU) so gradients flow
        more freely. Sigmoid is kept only for the gate (gating ∈ [0,1]).
    """
    def __init__(self, in_dim, norm, act):
        super().__init__()
        # Thin-edge detector: subtract 3×3 mean
        self.pool3 = nn.AvgPool2d(3, stride=1, padding=1)
        # Coarse-edge detector: subtract 5×5 mean
        self.pool5 = nn.AvgPool2d(5, stride=1, padding=2)

        # Fuse both edge maps into `in_dim` channels
        self.fuse = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, 1, bias=False),
            norm(in_dim),
            act()
        )
        # Channel-attention gate: selectively amplifies informative edge channels
        reduced = max(in_dim // 4, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dim, reduced, 1),
            act(),
            nn.Conv2d(reduced, in_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        e3   = x - self.pool3(x)                         # fine edges
        e5   = x - self.pool5(x)                         # coarser edges
        edge = self.fuse(torch.cat([e3, e5], dim=1))     # fused edge map
        w    = self.gate(edge)                           # channel-wise weight
        return x + edge * w                              # residual addition


class MAEM(nn.Module):
    """
    Multi-scale Edge Enhancement Module — improved parallel-scale version.

    Key improvements over the original:
    1. Parallel pooling (not sequential):
       Original applied the same pool repeatedly (mid = pool(pool(pool(x)))),
       so later branches were just blurrier versions of each other.
       New design applies pools of sizes {3,5,7} to the *same* base feature,
       giving genuinely independent scales of spatial context.

    2. Proper activations:
       Original used Sigmoid inside `in_conv` and `mid_conv`. Sigmoid saturates
       at extremes and kills gradients. We now use the passed `act` (ReLU/GELU)
       which is standard for feature extraction.

    3. All branches receive edge enhancement:
       Original skipped edge enhancement on the finest (first) branch.
       Now all `width` branches — including the base — go through EdgeEnhancer.

    4. Scale-attention (SE-block on concatenated branches):
       Learns which scale and which channel is most useful for the current
       feature map, rather than treating all scales identically before out_conv.
    """
    def __init__(self, in_dim, hidden_dim, width, norm, act):
        super().__init__()
        self.in_dim    = in_dim
        self.hidden_dim = hidden_dim
        self.width     = width

        # Shared projection: in_dim → hidden_dim (with proper activation)
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            norm(hidden_dim),
            act()
        )

        # Parallel average pools at increasing radii (width-1 extra branches)
        pool_sizes = [3, 5, 7, 9][: width - 1]
        self.pools = nn.ModuleList([
            nn.AvgPool2d(k, stride=1, padding=k // 2) for k in pool_sizes
        ])

        # Per-branch projection (with proper activation, not Sigmoid)
        self.mid_conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                norm(hidden_dim),
                act()
            )
            for _ in range(width - 1)
        ])

        # Edge enhancer applied to *every* branch (including base)
        self.edge_enhance = nn.ModuleList([
            EdgeEnhancer(hidden_dim, norm, act) for _ in range(width)
        ])

        # Scale attention: SE-block over all concatenated branches
        total_ch = hidden_dim * width
        reduced  = max(total_ch // 4, 1)
        self.scale_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_ch, reduced, 1),
            act(),
            nn.Conv2d(reduced, total_ch, 1),
            nn.Sigmoid()
        )

        # Output projection: hidden_dim*width → in_dim
        self.out_conv = nn.Sequential(
            nn.Conv2d(total_ch, in_dim, 1, bias=False),
            norm(in_dim),
            act()
        )

    def forward(self, x):
        base = self.in_conv(x)                        # (B, hidden_dim, H, W)

        # Branch 0: base features (no extra pooling)
        branches = [self.edge_enhance[0](base)]

        # Branches 1…width-1: apply parallel pools to the *same* base
        for i in range(self.width - 1):
            pooled = self.pools[i](base)              # scale-specific blur
            pooled = self.mid_conv[i](pooled)         # project
            pooled = self.edge_enhance[i + 1](pooled) # edge-enhance
            branches.append(pooled)

        out  = torch.cat(branches, dim=1)             # (B, hidden_dim*width, H, W)
        attn = self.scale_attn(out)                   # channel-wise scale weights
        out  = out * attn                             # reweight
        out  = self.out_conv(out)                     # (B, in_dim, H, W)
        return out


class BARM(nn.Module):
    """
    Detail Enhancement Module with FPN-based CNN feature integration.

    Changes from previous version:
      - Removed the single `img_in_conv` (3→img_dim conv on raw image) which
        was too weak to extract useful high-resolution features.
      - Added an FPN decoder that merges the three CNN encoder scales
        (s8=128ch@64, s4=64ch@128, s2=32ch@256) back up to img_dim@512.
        This gives the module proper multi-scale, stride-preserved image
        features instead of a single shallow conv output.

    Channel flow through FPN (all sizes for default img_size=512, img_dim=32):
      s8  (B,128, 64, 64) → fpn_s8_proj → (B, 64, 64, 64)
                          → upsample 2× → (B, 64,128,128)
                          → cat(s4=64)  → (B,128,128,128)
                          → fpn_merge_s4 → (B, 64,128,128)
                          → upsample 2× → (B, 64,256,256)
                          → cat(s2=32)  → (B, 96,256,256)
                          → fpn_merge_s2 → (B, 32,256,256)  [= img_dim]
                          → fpn_final_up → (B, 32,512,512)
      → MAEM residual     → (B, 32,512,512)

    CNN encoder output channels (32, 64, 128) are fixed by SPPE.
    `img_dim` is 32 in the BAHNet configuration.

    All downstream shapes (attn_gate, fusion_conv) are unchanged because
    the FPN final output is still img_dim=32, same as before.
    """
    def __init__(self, img_dim, feature_dim, norm, act):
        super().__init__()

        # --- FPN components (replacing img_in_conv) ---
        # CNN encoder outputs: s2=32ch, s4=64ch, s8=128ch  (fixed by SPPE)

        # Project s8 128→64
        self.fpn_s8_proj = nn.Sequential(
            nn.Conv2d(128, 64, 1, bias=False),
            norm(64),
            act(),
        )
        # Merge upsampled-s8 (64) + s4 (64) = 128 → 64
        self.fpn_merge_s4 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            norm(64),
            act(),
        )
        # Merge upsampled-s4-merged (64) + s2 (32) = 96 → img_dim (32)
        self.fpn_merge_s2 = nn.Sequential(
            nn.Conv2d(96, img_dim, 3, padding=1, bias=False),
            norm(img_dim),
            act(),
        )
        # Upsample merged-s2 256→512
        self.fpn_final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(img_dim, img_dim, 3, padding=1, bias=False),
            norm(img_dim),
            act(),
        )

        # MAEM on FPN output (retains multi-scale edge enhancement)
        self.maem = MAEM(img_dim, img_dim // 2, 4, norm, act)

        # Attention gate: decoder features select which CNN features matter.
        # Input: cat(decoder_feature=feature_dim, cnn_feature=img_dim) → img_dim gate
        self.attn_gate = nn.Sequential(
            nn.Conv2d(feature_dim + img_dim, img_dim, 1, bias=False),
            norm(img_dim),
            nn.Sigmoid()
        )

        # Fusion conv: feature_dim(32) + img_dim(32) = 64 → 64 → 32 → 16
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim + img_dim, 64, 3, padding=1, bias=False),
            norm(64),
            act(),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            norm(32),
            act(),
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            norm(16),
            act(),
        )

        self.out_conv = nn.Conv2d(16, 1, 1)

        # Upsample decoder+deep features from 128×128 → 512×512  (unchanged)
        self.feature_upsample = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            norm(feature_dim),
            act(),
        )

    def forward(self, cnn_feats, feature, b_feature):
        """
        Args:
            cnn_feats : tuple (s2, s4, s8) from SPPE
                          s2 (B,  32, H/2, W/2)
                          s4 (B,  64, H/4, W/4)
                          s8 (B, 128, H/8, W/8)
            feature   : (B, 32, H/4, W/4)  — MaskDecoder upscaled embedding
            b_feature : (B, 32, H/4, W/4)  — deep_feautre_conv output
        """
        s2, s4, s8 = cnn_feats

        # --- FPN top-down merge ---
        p8    = self.fpn_s8_proj(s8)                                              # (B, 64, H/8, W/8)
        p8_up = F.interpolate(p8, scale_factor=2, mode='bilinear', align_corners=False)  # (B, 64, H/4, W/4)
        p4    = self.fpn_merge_s4(torch.cat([p8_up, s4], dim=1))                  # (B, 64, H/4, W/4)
        p4_up = F.interpolate(p4, scale_factor=2, mode='bilinear', align_corners=False)  # (B, 64, H/2, W/2)
        p2    = self.fpn_merge_s2(torch.cat([p4_up, s2], dim=1))                  # (B, img_dim, H/2, W/2)
        img_feature = self.fpn_final_up(p2)                                       # (B, img_dim, H, W)

        # MAEM edge enhancement (residual)
        img_feature = self.maem(img_feature) + img_feature                      # (B, img_dim, H, W)

        # Upsample decoder features to full resolution  (unchanged)
        feature     = torch.cat([feature, b_feature], dim=1)                      # (B, feature_dim*2, H/4, W/4)
        feature     = self.feature_upsample(feature)                              # (B, feature_dim,   H, W)

        # Attention gate: decoder decides what CNN detail is relevant
        attn        = self.attn_gate(torch.cat([feature, img_feature], dim=1))    # (B, img_dim, H, W)
        img_feature = img_feature * attn

        # Fuse and predict
        out_feature = torch.cat([feature, img_feature], dim=1)
        out_feature = self.fusion_conv(out_feature)
        out         = self.out_conv(out_feature)
        return out


class HLIF(nn.Module):
    def __init__(self, norm, act):
        super().__init__()
        self.fusi_conv = nn.Sequential(
            nn.Conv2d(1024, 256, 1,bias = False),
            norm(256),
            act(),
        )

        self.context_conv = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False), norm(256), act(),
            nn.Conv2d(256, 256, 3, padding=1, groups=256, bias=False),
        )
        self.attn_conv = nn.ModuleList()
        for i in range(4):
            self.attn_conv.append(nn.Sequential(
                nn.Conv2d(256, 256, 1,bias = False),
                norm(256),
                act(),
            ))

        self.attn_conv_min = nn.ModuleList()
        for i in range(4):
                self.attn_conv_min.append(nn.Sequential(
                    nn.Conv2d(256, 256, 1,bias = False),
                    norm(256),
                    act(),
                ))

        # side heads (one per branch)
        self.side_heads = nn.ModuleList([nn.Conv2d(256, 1, kernel_size=1, bias=True) for _ in range(4)])

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.sigmoid = nn.Sigmoid()

        self.myact=act()

    def forward(self, feature_list):
        fusi_feature = torch.cat(feature_list, dim = 1).contiguous()
        fusi_feature = self.fusi_conv(fusi_feature)
        side_logits = []

        for i in range(4):
            x = feature_list[i]
            context = self.context_conv(fusi_feature)
            gate = self.sigmoid(self.pool(self.attn_conv[i](context)))
            reinforced = gate * x + x
            difference = context - x
            diff_gate = self.sigmoid(self.pool(self.attn_conv_min[i](difference)))
            redundant = diff_gate * difference + reinforced
            refined = self.myact(reinforced - redundant)
            feature_list[i] = refined
            side_logits.append(self.side_heads[i](refined))


        fused = feature_list[0] + feature_list[1] + feature_list[2] + feature_list[3]
        return fused, side_logits
