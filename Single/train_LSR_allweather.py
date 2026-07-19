"""Train LSR (ASR module) on a real paired weather dataset (raindrop / rain+fog / snow).

Compared with train_LSR_mix.py, this script feeds the LQ secret image from the
dataset directly instead of synthesising noise/blur/down-sample on the fly.

Training data layout (mixed, single root):

    --train_root <root>/
        input/   # degraded images (rain + raindrop + snow mixed)
        gt/

Testing data: three separately-rooted datasets with different subdir names.
Each test set is evaluated independently and metrics are logged per task:

    --rainhaze_test <root>/  contains data/ and gt/
    --raindrop_test <root>/  contains data/ and gt/
    --snow_test <root>/      contains synthetic/ and gt/

Any of the three test flags can be omitted to skip that test set.
"""

import os
import argparse
import shutil
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import PairedImageFolder
from losses import LSR_Loss
import logging
import numpy as np
import PIL.Image as Image
from torchvision.transforms import ToPILImage
from typing import Tuple, Union
from torch.utils.tensorboard import SummaryWriter
from util import DWT, IWT, setup_logger
from LIH import LIH as hide_model
from LSR import Model as restore_model
from tqdm import tqdm
import lpips
import metrics as my_metrics


def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()
    return noise


def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.cpu().clamp_(0, 1).squeeze())


def _to_bgr_uint8(x: Union[torch.Tensor, np.ndarray, Image.Image]) -> np.ndarray:
    """Normalise an image of any common type to BGR uint8 HWC (cv2 convention).

    `metrics.py` was adapted from SwinIR and uses `bgr2ycbcr` whose coefficients
    expect BGR input. Our pipeline produces RGB tensors / PIL images, so we
    flip channels here.
    """
    if isinstance(x, Image.Image):
        rgb = np.asarray(x)
    elif isinstance(x, torch.Tensor):
        t = x.detach().cpu().clamp(0, 1).squeeze()
        if t.dim() == 3:  # CHW
            t = t.permute(1, 2, 0)
        rgb = (t.numpy() * 255.0).round().astype(np.uint8)
    elif isinstance(x, np.ndarray):
        rgb = x
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255.0).round().astype(np.uint8)
    else:
        raise TypeError(f"Unsupported image type: {type(x)}")
    return np.ascontiguousarray(rgb[:, :, ::-1])  # RGB → BGR


def compute_metrics(a, b, y_channel: bool = True) -> Tuple[float, float]:
    """PSNR + SSIM via metrics.py (SwinIR-style).

    Args:
        a, b: tensor / ndarray / PIL Image in RGB.
        y_channel: True → compute on Y of YCbCr (standard for weather
            restoration baselines); False → compute on RGB.
    """
    a_bgr = _to_bgr_uint8(a)
    b_bgr = _to_bgr_uint8(b)
    p = my_metrics.calculate_psnr(a_bgr, b_bgr, test_y_channel=y_channel)
    s = my_metrics.calculate_ssim(a_bgr, b_bgr, test_y_channel=y_channel)
    return p, s


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum_sq = 0
        self.count = 0
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.sum_sq += (val ** 2) * n
        self.count += n
        self.avg = self.sum / self.count
        var = self.sum_sq / self.count - self.avg ** 2
        self.std = max(var, 0) ** 0.5


class CustomDataParallel(nn.DataParallel):
    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    params_dict = dict(net.named_parameters())
    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    return optimizer


def split_batch(input_lq, gt):
    """Split a paired batch into cover/secret halves.

    Returns:
        cover_img: clean carrier (from gt of second half)
        secret_gt: HQ ground truth of the secret (loss target)
        mix_secret: LQ degraded secret to be hidden (real weather input)
    """
    B = gt.shape[0]
    half = B // 2
    cover_img = gt[half:]
    secret_gt = gt[:half]
    mix_secret = input_lq[:half]
    return cover_img, secret_gt, mix_secret


def train_one_epoch(
    hide_model, denoise_model, criterion, train_dataloader,
    hide_optimizer, denoise_optimizer, epoch, logger_train, tb_logger, args,
):
    if args.finetune:
        hide_model.train()
    else:
        for param in hide_model.parameters():
            param.requires_grad = False
    denoise_model.train()

    device = next(hide_model.parameters()).device
    dwt = DWT()
    iwt = IWT()

    for i, (input_lq, gt) in enumerate(train_dataloader):
        input_lq = input_lq.to(device)
        gt = gt.to(device)
        if gt.shape[0] < 2:
            continue  # need at least 2 samples to split into cover/secret

        cover_img, secret_img, mix_secret = split_batch(input_lq, gt)

        input_cover = dwt(cover_img)
        input_secret = dwt(mix_secret)

        if args.finetune:
            hide_optimizer.zero_grad()
        denoise_optimizer.zero_grad()

        # hide
        output_steg, output_z = hide_model(input_cover, input_secret)
        steg_img = iwt(output_steg)

        # secure restoration on stego
        steg_clean = denoise_model(steg_img)
        output_clean = dwt(steg_clean)

        # reveal
        output_z_guass = gauss_noise(output_z.shape)
        cover_rev, secret_rev = hide_model(output_clean, output_z_guass, rev=True)
        rec_img = iwt(secret_rev)

        out_criterion = criterion(
            secret_img, cover_img, steg_clean, steg_img, rec_img,
            args.sweight, args.cweight, args.pweight_c, args.finetune,
        )
        loss = out_criterion["loss"]
        loss.backward()
        denoise_optimizer.step()
        if args.finetune:
            hide_optimizer.step()

        if i % 10 == 0:
            logger_train.info(
                f"Train epoch {epoch}: ["
                f"{i*gt.shape[0]}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f"\tLoss: {loss.item():.3f} |"
            )
    tb_logger.add_scalar("[train]: loss", loss.item(), epoch)


def test_epoch(args, epoch, test_dataloader, hide_model, denoise_model,
               logger_val, criterion, lpips_fn, task_name="test"):
    dwt = DWT()
    iwt = IWT()
    hide_model.eval()
    denoise_model.eval()
    device = next(hide_model.parameters()).device

    psnrc = AverageMeter(); ssimc = AverageMeter()
    psnrs = AverageMeter(); ssims = AverageMeter()
    psnrcori = AverageMeter(); ssimcori = AverageMeter()
    lpipsc = AverageMeter(); lpipss = AverageMeter(); lpipscori = AverageMeter()
    loss = AverageMeter()
    i = 0

    with torch.no_grad():
        for idx, (input_lq, gt) in tqdm(enumerate(test_dataloader)):
            input_lq = input_lq.to(device)
            gt = gt.to(device)
            if gt.shape[0] < 2:
                continue

            cover_img, secret_img, input_secret_img = split_batch(input_lq, gt)

            input_cover = dwt(cover_img)
            input_secret = dwt(input_secret_img)

            output_steg, output_z = hide_model(input_cover, input_secret)
            steg_ori = iwt(output_steg)

            steg_clean = denoise_model(steg_ori)
            output_clean = dwt(steg_clean)

            output_z_guass = gauss_noise(output_z.shape)
            cover_rev, secret_rev = hide_model(output_clean, output_z_guass, rev=True)
            rec_img = iwt(secret_rev)

            out_criterion = criterion(
                secret_img, cover_img, steg_clean, steg_ori, rec_img,
                args.sweight, args.cweight, args.pweight_c, args.finetune,
            )
            loss.update(out_criterion["loss"])

            lc = lpips_fn.forward(cover_img, steg_clean)
            ls = lpips_fn.forward(secret_img, rec_img)
            lori = lpips_fn.forward(cover_img, steg_ori)
            lpipsc.update(lc.mean().item())
            lpipss.update(ls.mean().item())
            lpipscori.update(lori.mean().item())

            save_dir = os.path.join("experiments", args.experiment, "images")

            secret_img_pil = torch2img(secret_img)
            cover_img_pil = torch2img(cover_img)
            degrade_secret_img_pil = torch2img(input_secret_img)
            secret_img_rec_pil = torch2img(rec_img)
            steg_img_pil = torch2img(steg_clean)
            steg_img_ori_pil = torch2img(steg_ori)

            # All metrics on Y channel of YCbCr.
            p1, m1 = compute_metrics(secret_img_rec_pil, secret_img_pil, y_channel=True)
            psnrs.update(p1); ssims.update(m1)
            p2, m2 = compute_metrics(steg_img_pil, cover_img_pil, y_channel=True)
            psnrc.update(p2); ssimc.update(m2)
            p3, m3 = compute_metrics(steg_img_ori_pil, cover_img_pil, y_channel=True)
            psnrcori.update(p3); ssimcori.update(m3)

            if args.save_img and (args.save_img_limit <= 0 or i < args.save_img_limit):
                rec_dir = os.path.join(save_dir, "rec", task_name)
                secret_dir = os.path.join(save_dir, "secret", task_name)
                cover_dir = os.path.join(save_dir, "cover", task_name)
                stego_dir = os.path.join(save_dir, "stego", task_name)
                ori_stego_dir = os.path.join(save_dir, "ori_stego", task_name)
                for d in (rec_dir, secret_dir, cover_dir, stego_dir, ori_stego_dir):
                    os.makedirs(d, exist_ok=True)

                steg_img_pil.save(os.path.join(stego_dir, "%03d.png" % i))
                secret_img_rec_pil.save(os.path.join(rec_dir, "%03d.png" % i))
                degrade_secret_img_pil.save(os.path.join(secret_dir, "%03d.png" % i))
                cover_img_pil.save(os.path.join(cover_dir, "%03d.png" % i))
                steg_img_ori_pil.save(os.path.join(ori_stego_dir, "%03d.png" % i))
                i += 1

    logger_val.info(
        f"Test epoch {epoch} - [{task_name}]: Average metrics:"
        f"\tPSNRC: {psnrc.avg:.6f}±{psnrc.std:.6f} |"
        f"\tSSIMC: {ssimc.avg:.6f}±{ssimc.std:.6f} |"
        f"\tLPIPSC: {lpipsc.avg:.6f}±{lpipsc.std:.6f} |"
        f"\tPSNRS: {psnrs.avg:.6f}±{psnrs.std:.6f} |"
        f"\tSSIMS: {ssims.avg:.6f}±{ssims.std:.6f} |"
        f"\tLPIPSS: {lpipss.avg:.6f}±{lpipss.std:.6f} |"
        f"\tPSNRCORI: {psnrcori.avg:.6f}±{psnrcori.std:.6f} |"
        f"\tSSIMCORI: {ssimcori.avg:.6f}±{ssimcori.std:.6f} |"
        f"\tLPIPSORI: {lpipscori.avg:.6f}±{lpipscori.std:.6f} |\n"
    )

    return loss.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        dest_filename = filename.replace(filename.split("/")[-1], "_checkpoint_best_loss.pth.tar")
        shutil.copyfile(filename, dest_filename)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Train LSR on paired all-weather dataset.")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="Training dataset root (must contain input/ and gt/)")
    parser.add_argument("--rainhaze_test", type=str, default=None,
                        help="Rain+haze test root (expects data/ and gt/)")
    parser.add_argument("--raindrop_test", type=str, default=None,
                        help="Raindrop test root (expects data/ and gt/)")
    parser.add_argument("--snow_test", type=str, default=None,
                        help="Snow test root (expects synthetic/ and gt/)")
    parser.add_argument("-e", "--epochs", default=100000, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float)
    parser.add_argument("-n", "--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=2)
    parser.add_argument("--patch-size", type=int, nargs=2, default=(224, 224))
    parser.add_argument("--test-patch-size", type=int, nargs=2, default=(256, 256))
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--save", action="store_true", default=True)
    parser.add_argument("--hide_checkpoint", type=str, required=True,
                        help="Path to a LIH checkpoint trained on clean gt images")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("-exp", "--experiment", type=str, required=True)
    parser.add_argument("--val_freq", type=int, default=30)
    parser.add_argument("--channels_in", type=int, default=12)
    parser.add_argument("--num_step", type=int, default=12)
    parser.add_argument("--klvl", type=int, default=3)
    parser.add_argument("--mid", type=int, default=2)
    parser.add_argument("--enc", default=[2, 2, 4], nargs="+", type=int)
    parser.add_argument("--dec", default=[2, 2, 2], nargs="+", type=int)
    parser.add_argument("--save_img", action="store_true", default=False)
    parser.add_argument("--save_img_limit", type=int, default=0,
                        help="Cap saved images per test set (0 = unlimited).")
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--finetune", action="store_true", default=False)
    parser.add_argument("--sweight", type=float, default=2)
    parser.add_argument("--cweight", type=float, default=1)
    parser.add_argument("--pweight_c", type=float, default=0.01)
    parser.add_argument("--lfrestore", type=bool, nargs="?", const=True, default=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--nafwidth", default=32, type=int)
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    exp_dir = os.path.join("experiments", args.experiment)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)

    setup_logger("train", exp_dir, "train_" + args.experiment,
                 level=logging.INFO, screen=True, tofile=True)
    setup_logger("val", exp_dir, "val_" + args.experiment,
                 level=logging.INFO, screen=True, tofile=True)
    logger_train = logging.getLogger("train")
    logger_val = logging.getLogger("val")
    tb_logger = SummaryWriter(log_dir="./tb_logger/" + args.experiment)

    train_dataset = PairedImageFolder(
        args.dataset, patch_size=tuple(args.patch_size), split="train",
        input_subdir="input", gt_subdir="gt",
    )

    test_specs = [
        ("rainhaze", args.rainhaze_test, "data", "gt"),
        ("raindrop", args.raindrop_test, "data", "gt"),
        ("snow",     args.snow_test,     "synthetic", "gt"),
    ]
    test_loaders = []  # list of (task_name, DataLoader)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, pin_memory=(device == "cuda"), drop_last=True,
    )

    for name, root, in_sub, gt_sub in test_specs:
        if root is None:
            continue
        ds = PairedImageFolder(
            root, patch_size=tuple(args.test_patch_size), split="test",
            input_subdir=in_sub, gt_subdir=gt_sub,
        )
        loader = DataLoader(
            ds, batch_size=args.test_batch_size, num_workers=args.num_workers,
            shuffle=False, pin_memory=(device == "cuda"), drop_last=True,
        )
        test_loaders.append((name, loader))

    if len(test_loaders) == 0:
        raise RuntimeError(
            "No test datasets provided. Pass at least one of "
            "--rainhaze_test / --raindrop_test / --snow_test."
        )

    hide_net = hide_model(args.num_step).to(device)
    denoise_net = restore_model(
        steps=args.steps, klvl=args.klvl, mid=args.mid,
        enc=args.enc, dec=args.dec, lfrestore=args.lfrestore, width=args.nafwidth,
    ).to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        denoise_net = CustomDataParallel(denoise_net)
    logger_train.info(args)

    state_dicts = torch.load(args.hide_checkpoint, map_location=device)
    hide_net.load_state_dict(state_dicts["state_dict"])

    hide_optimizer = configure_optimizers(hide_net, args)
    denoise_optimizer = configure_optimizers(denoise_net, args)

    criterion = LSR_Loss()
    lpips_fn = lpips.LPIPS(net="alex", version="0.1")
    if device == "cuda":
        lpips_fn.cuda()

    last_epoch = 0
    loss = float("inf")
    best_loss = float("inf")
    if args.checkpoint:
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        denoise_net.load_state_dict(checkpoint["state_dict"])
        denoise_optimizer.load_state_dict(checkpoint["optimizer"])
        denoise_optimizer.param_groups[0]["lr"] = args.learning_rate

    if not args.test:
        for epoch in range(last_epoch, args.epochs):
            logger_train.info(f"Learning rate: {denoise_optimizer.param_groups[0]['lr']}")
            train_one_epoch(
                hide_net, denoise_net, criterion, train_dataloader,
                hide_optimizer, denoise_optimizer, epoch, logger_train, tb_logger, args,
            )
            if epoch % args.val_freq == 0:
                losses = []
                for name, loader in test_loaders:
                    l = test_epoch(
                        args, epoch, loader, hide_net, denoise_net,
                        logger_val, criterion, lpips_fn, task_name=name,
                    )
                    losses.append(l)
                loss = float(sum(losses) / len(losses))

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            if args.save:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": denoise_net.state_dict(),
                        "best_loss": best_loss,
                        "optimizer": denoise_optimizer.state_dict(),
                    },
                    is_best,
                    os.path.join(exp_dir, "checkpoints", "net_checkpoint.pth.tar"),
                )
                if args.finetune:
                    save_folder = os.path.join(exp_dir, "hide_checkpoints")
                    os.makedirs(save_folder, exist_ok=True)
                    save_checkpoint(
                        {
                            "epoch": epoch,
                            "state_dict": hide_net.state_dict(),
                            "best_loss": best_loss,
                            "optimizer": hide_optimizer.state_dict(),
                        },
                        is_best,
                        os.path.join(save_folder, "net_checkpoint.pth.tar"),
                    )
                if is_best:
                    logger_val.info("best checkpoint saved.")
    else:
        for name, loader in test_loaders:
            test_epoch(
                args, 0, loader, hide_net, denoise_net,
                logger_val, criterion, lpips_fn, task_name=name,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
