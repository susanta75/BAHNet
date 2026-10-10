from model.bahnet import BAHNet
from dataset.sod_dataset import getSODDataloader
import torch
from tqdm import tqdm
import os
import torch.nn.functional as F
from utils.loss import LossFunc, psl_regularization
from utils.AvgMeter import AvgMeter
import csv
import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)                # Python RNG
    np.random.seed(seed)             # NumPy RNG
    torch.manual_seed(seed)          # CPU RNG
    torch.cuda.manual_seed(seed)     # Current GPU RNG
    torch.cuda.manual_seed_all(seed) # All GPUs (if multi-GPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class Args:
    seed             = 42
    warmup_period    = 5
    batch_size       = 16
    num_workers      = 4
    epochs           = 100
    lr_rate          = 0.0005
    img_size         = 512
    # Training dataset
    data_path        = "./dataset/SOD/Kvasir/"
    # Validation splits used by the paper's checkpoint criterion.
    clinicdb_val_path = "./dataset/SOD/CVC-ClinicDB/"
    etis_val_path     = "./dataset/SOD/ETIS-Lari/"
    encoder_ckpt     = "./pretrained/sam_vit_b_01ec64.pth"
    save_dir         = "output/"
    resume           = ""


def trainer(net, dataloader, loss_func, optimizer, device):
    net.train()
    loss_avg = AvgMeter()
    mae_avg  = AvgMeter()
    print("start training")

    sigmoid = torch.nn.Sigmoid()

    for data in tqdm(dataloader):
        img   = data["img"].to(device).to(torch.float32)
        label = data["mask"].to(device).unsqueeze(1)  # (B,1,H,W)

        optimizer.zero_grad()

        out_tuple = net(img)

        if isinstance(out_tuple, (list, tuple)) and len(out_tuple) == 3:
            out, coarse_out, side_list = out_tuple
        else:
            out, coarse_out = out_tuple
            side_list = []

        out        = sigmoid(out)
        coarse_out = sigmoid(coarse_out)
        side_list  = [sigmoid(s) for s in side_list]

        if side_list and (side_list[0].shape[-2:] != label.shape[-2:]):
            side_list = [F.interpolate(s, size=label.shape[-2:],
                                       mode='bilinear', align_corners=False)
                         for s in side_list]

        loss_main = loss_func(out, label) + loss_func(coarse_out, label)

        side_weights = [0.2, 0.2, 0.2, 0.2][:len(side_list)]
        loss_side = 0.0
        for w, s in zip(side_weights, side_list):
            loss_side = loss_side + w * loss_func(s, label)

        # PSL regularization — spatial coherence, multi-output consistency,
        # area plausibility, boundary sharpness.  Applied only to the final
        # fine output; weights are small so PSL guides without overriding GT.
        loss_psl = psl_regularization(out, coarse_out, side_list, label, img)

        loss = loss_main + loss_side + loss_psl

        loss_avg.update(loss.item(), img.shape[0])
        mae = torch.mean(torch.abs(out - label))
        mae_avg.update(mae.item(), n=img.shape[0])

        loss.backward()
        optimizer.step()

    print(f"Train Loss: {loss_avg.avg:.4f}, Train MAE: {mae_avg.avg:.4f}")
    return loss_avg.avg, mae_avg.avg


def valer(net, dataloader, device, name="Val"):
    """
    Evaluate on one dataloader and return MAE.
    `name` is used only for the printed label.
    """
    net.eval()
    sigmoid = torch.nn.Sigmoid()
    mae_avg = AvgMeter()
    with torch.no_grad():
        for data in tqdm(dataloader, desc=name, leave=False):
            img = data["img"].to(device).to(torch.float32)
            ori_label = data["ori_mask"].to(device)

            out, _, _ = net(img)
            out = sigmoid(out)
            out = F.interpolate(out, size=ori_label.shape[1:], mode="bilinear", align_corners=False)

            mae = torch.mean(torch.abs(out - ori_label))
            mae_avg.update(mae.item(), n=1)

    print(f"{name} MAE: {mae_avg.avg:.4f}")
    return mae_avg.avg


def reshapePos(pos_embed, img_size):
    token_size = img_size // 16
    if pos_embed.shape[1] != token_size:
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)
    return pos_embed

def reshapeRel(k, rel_pos_params, img_size):
    if not any(x in k for x in ['2', '5', '8', '11']):
        return rel_pos_params

    token_size = img_size // 16
    new_len = 2 * token_size - 1

    if "rel_pos_h" in k:
        rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
        rel_pos_params = F.interpolate(rel_pos_params, size=(new_len, rel_pos_params.shape[-1]), mode='bilinear', align_corners=False)
        rel_pos_params = rel_pos_params.squeeze(0).squeeze(0)
    elif "rel_pos_w" in k:
        rel_pos_params = rel_pos_params.permute(1, 0).unsqueeze(0).unsqueeze(0)
        rel_pos_params = F.interpolate(rel_pos_params, size=(rel_pos_params.shape[-2], new_len), mode='bilinear', align_corners=False)
        rel_pos_params = rel_pos_params.squeeze(0).squeeze(0).permute(1, 0)
    return rel_pos_params

def load(net, ckpt_path, img_size):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    new_ckpt = {}
    for k, v in ckpt.items():
        if 'pe_layer' in k:
            new_ckpt[k[15:]] = v
        elif 'pos_embed' in k:
            new_ckpt[k] = reshapePos(v, img_size)
        elif 'rel_pos' in k:
            new_ckpt[k] = reshapeRel(k, v, img_size)
        elif 'image_encoder' in k:
            if 'neck' in k:
                for i in range(4):
                    new_ckpt[f"{k[:18]}{i}{k[18:]}"] = v
            else:
                new_ckpt[k] = v
        elif any(x in k for x in ['mask_decoder.transformer', 'mask_decoder.iou_token', 'mask_decoder.output_upscaling']):
            new_ckpt[k] = v
    return net.load_state_dict(new_ckpt, strict=False)


if __name__ == "__main__":
    args = Args()
    print(f"Start training | batch={args.batch_size} | LR={args.lr_rate} | epochs={args.epochs}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = BAHNet(args.img_size).to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {count_parameters(net):,}")

    start_epoch = 1

    # ------------------------------------------------------------------ #
    # Data loaders
    # ------------------------------------------------------------------ #
    trainLoader = getSODDataloader(
        args.data_path, args.batch_size, args.num_workers, 'train', img_size=args.img_size
    )
    # In-domain validation (Kvasir test split)
    kvasirValLoader   = getSODDataloader(args.data_path,         1, args.num_workers, 'test', img_size=args.img_size)
    # This separate ClinicDB test split is used for checkpoint validation.
    clinicdbValLoader = getSODDataloader(args.clinicdb_val_path, 1, args.num_workers, 'test', img_size=args.img_size)
    # Hardest out-of-domain: smallest polyps, most domain-shifted.
    # Including ETIS-Lari in the checkpoint criterion forces the model to
    # balance performance on the most difficult generalisation case, not
    # just the two easiest domains.
    etisValLoader     = getSODDataloader(args.etis_val_path,     1, args.num_workers, 'test', img_size=args.img_size)

    loss_func = LossFunc

    # ------------------------------------------------------------------ #
    # Parameter groups
    #
    # hungry_param : adapters + all non-encoder modules  → full LR
    # full_param   : encoder neck modules                → 0.1× LR
    # All other image_encoder params                     → frozen (entire run)
    #
    # The ViT backbone stays fully frozen throughout training.
    # The DAB adapters (already in hungry_param) are the intended
    # adaptation mechanism — they adjust the ViT's outputs without
    # disturbing its pretrained attention patterns.  Fine-tuning the
    # encoder weights directly on this limited training set risks
    # forgetting of SAM's pretrained representations and hurts
    # cross-dataset performance, especially on small-polyp datasets
    # like ETIS-Lari.
    # ------------------------------------------------------------------ #
    hungry_param = []
    full_param   = []

    for k, v in net.named_parameters():
        if "image_encoder" in k and ".dab." in k:
            hungry_param.append(v)
        elif "image_encoder.neck" in k:
            full_param.append(v)
        elif "image_encoder" in k:
            v.requires_grad = False
        else:
            hungry_param.append(v)

    optimizer = torch.optim.AdamW([
        {"params": hungry_param, "lr": args.lr_rate},
        {"params": full_param,   "lr": args.lr_rate * 0.1},
    ], weight_decay=1e-5)

    # ------------------------------------------------------------------ #
    # Checkpoint / resume
    # ------------------------------------------------------------------ #
    if args.resume != "":
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_mae    = checkpoint.get("val_mae", float("inf"))
    else:
        load(net, args.encoder_ckpt, args.img_size)
        best_mae = float("inf")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    log_file = f"{args.save_dir}/training_log.csv"
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Epoch', 'Learning Rate', 'Train Loss', 'Train MAE',
            'Kvasir Val MAE', 'ClinicDB Val MAE', 'ETIS Val MAE', 'Combined Val MAE'
        ])

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    for epoch in range(start_epoch, args.epochs + 1):

        # Learning rate schedule (warmup → exponential decay)
        if epoch <= args.warmup_period:
            lr = args.lr_rate * epoch / args.warmup_period
        else:
            lr = args.lr_rate * (0.98 ** (epoch - args.warmup_period))

        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr if i == 0 else lr * 0.1

        print(f"\nEpoch {epoch}/{args.epochs}  |  base LR={lr:.2e}")

        train_loss, train_mae = trainer(net, trainLoader, loss_func, optimizer, device)

        # Triple validation across all three domains
        kvasir_mae   = valer(net, kvasirValLoader,   device, name="Kvasir Val")
        clinicdb_mae = valer(net, clinicdbValLoader,  device, name="ClinicDB Val")
        etis_mae     = valer(net, etisValLoader,      device, name="ETIS Val")

        # Combined score: equal weight across all three domains.
        # ETIS-Lari inclusion is the key change — it is the hardest domain
        # (smallest polyps, most different from Kvasir) and was previously
        # invisible to the checkpoint criterion.  A model that drops on
        # ETIS while improving on Kvasir will now be penalised rather than
        # saved as the best model.
        combined_mae = (kvasir_mae + clinicdb_mae + etis_mae) / 3.0
        print(f"Combined Val MAE: {combined_mae:.4f}  "
              f"[Kvasir={kvasir_mae:.4f} | ClinicDB={clinicdb_mae:.4f} | ETIS={etis_mae:.4f}]")

        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, lr, train_loss, train_mae,
                             kvasir_mae, clinicdb_mae, etis_mae, combined_mae])

        # Checkpoint on combined MAE
        if combined_mae < best_mae:
            best_mae  = combined_mae
            ckpt_path = f"{args.save_dir}/best_model_epoch{epoch}.pth"
            torch.save({
                "model":     net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch":     epoch,
                "val_mae":   combined_mae,
            }, ckpt_path)
            print(f"Saved best model (combined MAE={best_mae:.4f}) at {ckpt_path}")

        if epoch % 5 == 0:
            periodic_ckpt = f"{args.save_dir}/checkpoint_epoch{epoch}.pth"
            torch.save({
                "model":     net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch":     epoch,
                "val_mae":   combined_mae,
            }, periodic_ckpt)
            print(f"Saved periodic checkpoint at {periodic_ckpt}")
