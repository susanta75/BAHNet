from dataset.sod_dataset import getSODDataloader
from model.bahnet import BAHNet
import torch
from tqdm import tqdm
import os
import shutil
from collections import OrderedDict
import numpy as np
import cv2
import py_sod_metrics
import csv

def Fmeasure_calu(pred, gt, threshold):
    if threshold > 1:
        threshold = 1

    Label3 = np.zeros_like(gt)
    Label3[pred >= threshold] = 1

    NumRec = np.sum(Label3 == 1)
    NumNoRec = np.sum(Label3 == 0)

    LabelAnd = (Label3 == 1) & (gt == 1)
    NumAnd = np.sum(LabelAnd == 1)
    num_obj = np.sum(gt)
    num_pred = np.sum(Label3)

    FN = num_obj - NumAnd
    FP = NumRec - NumAnd
    TN = NumNoRec - FN

    if NumAnd == 0:
        PreFtem = 0
        RecallFtem = 0
        FmeasureF = 0
        Dice = 0
        SpecifTem = 0
        IoU = 0
    else:
        IoU = NumAnd / (FN + NumRec + 1e-8)
        PreFtem = NumAnd / (NumRec + 1e-8)
        RecallFtem = NumAnd / (num_obj + 1e-8)
        SpecifTem = TN / (TN + FP + 1e-8)
        Dice = 2 * NumAnd / (num_obj + num_pred + 1e-8)
        FmeasureF = ((2.0 * PreFtem * RecallFtem) / (PreFtem + RecallFtem + 1e-8))

    return PreFtem, RecallFtem, SpecifTem, Dice, FmeasureF, IoU

def eval(net, dataloader, output_path, dataset, device, csv_writer=None):
    net.eval()
    print("start eval dataset:{}".format(dataset))

    sigmoid = torch.nn.Sigmoid()

    MAE = py_sod_metrics.MAE()
    WFM = py_sod_metrics.WeightedFmeasure()
    SM = py_sod_metrics.Smeasure()
    EM = py_sod_metrics.Emeasure()
    FM = py_sod_metrics.Fmeasure()

    total_dice = 0.0
    total_iou = 0.0
    valid_count = 0

    with torch.no_grad():
        for data in tqdm(dataloader, ncols=100):
            img = data["img"].to(device).to(torch.float32)
            ori_label = data['ori_mask']
            name = data['mask_name']

            out, _, _ = net(img)
            out = sigmoid(out)
            out = torch.nn.functional.interpolate(out, [ori_label.shape[1], ori_label.shape[2]], mode='bilinear', align_corners=False)


            #for Roc Curve and PR Curve calculation, we need to convert to uint8 [0,255]

            # Save probability map for ROC/PR curve
            # prob_map = out.squeeze().cpu().numpy()  # Float, [0,1]
            # filename = os.path.splitext(os.path.basename(name[0]))[0]
            # np.save(os.path.join(output_path, f"{filename}.npy"), prob_map)



            pred = (out * 255).squeeze().cpu().data.numpy().astype(np.uint8)
            ori_label = (ori_label * 255).squeeze(0).data.numpy().astype(np.uint8)

            # Normalize to [0,1] float for Dice/IoU
            pred_float = pred / 255.0
            gt_float = ori_label / 255.0

            FM.step(pred=pred, gt=ori_label)
            WFM.step(pred=pred, gt=ori_label)
            SM.step(pred=pred, gt=ori_label)
            EM.step(pred=pred, gt=ori_label)
            MAE.step(pred=pred, gt=ori_label)

            # Calculate Dice and IoU using Fmeasure_calu
            _, _, _, dice, _, iou = Fmeasure_calu(pred_float, gt_float, threshold=0.5)
            total_dice += dice
            total_iou += iou
            valid_count += 1

            pred_vis = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

            # Ensure the directory exists
            os.makedirs(output_path, exist_ok=True)

            # Clean filename
            filename = os.path.basename(name[0])
            save_path = os.path.join(output_path, filename)

            # Write and check
            if not cv2.imwrite(save_path, pred_vis):
               print(f"[Warning] Failed to save: {save_path}")


    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = MAE.get_results()["mae"]
    maxFm = FM.get_results()['mf']
    meanFm = fm['curve'].mean()
    em_mean = em['curve'].mean()

    mDice = total_dice / (valid_count + 1e-8)
    mIoU = total_iou / (valid_count + 1e-8)

    print("{} results:".format(dataset))
    print("mae:{:.3f}, maxFm:{:.3f}, sm:{:.3f}, em:{:.3f}, mDice:{:.3f}, mIoU:{:.3f}".format(
        mae, maxFm, sm, em_mean, mDice, mIoU))

    if csv_writer is not None:
        csv_writer.writerow([dataset,
                             f"{mae:.4f}",
                             f"{maxFm:.4f}",
                             f"{sm:.4f}",
                             f"{em_mean:.4f}",
                             f"{mDice:.4f}",
                             f"{mIoU:.4f}"])
    # ------------------------------------------------


class Args:
    checkpoint = "./output/best_model_epoch67.pth"
    # checkpoint = "./output/checkpoint_epoch75.pth"
    data_path = "./dataset/SOD/"
    result_path = "./results"
    num_workers = 4
    img_size = 512
    gpu_id = 0


if __name__ == "__main__":
    args = Args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    net = BAHNet(args.img_size).to(device)

    # Load pretrained weights
    ckpt_dic = torch.load(args.checkpoint, map_location=device)
    if 'model' in ckpt_dic.keys():
        ckpt_dic = ckpt_dic['model']
    dic = OrderedDict()
    for k, v in ckpt_dic.items():
        if 'module.' in k:
            dic[k[7:]] = v
        else:
            dic[k] = v
    msg = net.load_state_dict(dic, strict=False)
    print(msg)

    datasets = ["CVC-ClinicDB", "CVC-ColonDB", "ETIS-Lari", "Kvasir" ]
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    # --- CSV setup (added) ---
    csv_path = os.path.join(args.result_path, "sod_metrics.csv")
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    csv_file = open(csv_path, mode='a', newline='')
    csv_writer = csv.writer(csv_file)
    if write_header:
        csv_writer.writerow(["Dataset", "MAE", "MaxF", "S-measure", "E-measure", "mDice", "mIoU"])
    # --------------------------

    for dataset in datasets:
        testLoader = getSODDataloader(
            os.path.join(args.data_path, dataset),
            1,
            args.num_workers,
            'test',
            args.img_size
        )


        dataset_result_path = os.path.join(args.result_path, dataset)
        if os.path.exists(dataset_result_path):
            shutil.rmtree(dataset_result_path)
        os.makedirs(dataset_result_path)

        eval(net, testLoader, dataset_result_path, dataset, device, csv_writer)

    # --- Close CSV (added) ---
    if csv_file is not None:
        csv_file.close()
    print(f"Saved results CSV to: {csv_path}")
    # --------------------------
