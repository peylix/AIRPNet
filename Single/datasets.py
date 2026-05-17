# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class ImageFolder(Dataset):
    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img
   

    def __len__(self):
        return len(self.samples)


class PairedImageFolder(Dataset):
    """Load paired (input, gt) images from <root>/<input_subdir> and <root>/<gt_subdir>.

    Pairing strategy:
        1. Try matching by file stem (basename without extension).
        2. If no stems match but both directories contain the same number of
           files, fall back to sorted-order pairing (needed for datasets like
           raindrop where input is `<i>_rain.png` and gt is `<i>_clean.png`).

    Random crops applied during training are synchronized across the input/gt
    pair so the LQ image and its ground-truth stay aligned.
    """

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(self, root, patch_size=None, split="train",
                 input_subdir="input", gt_subdir="gt"):
        root = Path(root)
        self.input_dir = root / input_subdir
        self.gt_dir = root / gt_subdir
        if not self.input_dir.is_dir():
            raise RuntimeError(f'Invalid directory "{self.input_dir}"')
        if not self.gt_dir.is_dir():
            raise RuntimeError(f'Invalid directory "{self.gt_dir}"')

        input_files = sorted(
            f for f in self.input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.IMG_EXTS
        )
        gt_files = sorted(
            f for f in self.gt_dir.iterdir()
            if f.is_file() and f.suffix.lower() in self.IMG_EXTS
        )

        gt_by_stem = {f.stem: f for f in gt_files}
        pairs_by_stem = [(f, gt_by_stem[f.stem]) for f in input_files if f.stem in gt_by_stem]

        if len(pairs_by_stem) > 0:
            self.pairs = pairs_by_stem
        elif len(input_files) == len(gt_files) and len(input_files) > 0:
            self.pairs = list(zip(input_files, gt_files))
            print(
                f"[PairedImageFolder] No stem-matched pairs in '{root}'; "
                f"falling back to sorted-order pairing ({len(self.pairs)} pairs)."
            )
        else:
            raise RuntimeError(
                f'No matched (input, gt) pairs under "{root}" '
                f'(input={len(input_files)} files in "{input_subdir}/", '
                f'gt={len(gt_files)} files in "{gt_subdir}/"). '
                f'Either share file stems or have equal file counts.'
            )

        self.patch_size = patch_size
        self.split = split

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_path, gt_path = self.pairs[idx]
        input_img = Image.open(input_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        if self.patch_size is not None:
            ph, pw = self.patch_size
            w, h = input_img.size

            # If the image is smaller than the crop, upscale enough to crop.
            if h < ph or w < pw:
                new_h, new_w = max(h, ph), max(w, pw)
                input_img = TF.resize(input_img, [new_h, new_w])
                gt_img = TF.resize(gt_img, [new_h, new_w])
                w, h = input_img.size

            if self.split == "train":
                i = random.randint(0, h - ph)
                j = random.randint(0, w - pw)
                input_img = TF.crop(input_img, i, j, ph, pw)
                gt_img = TF.crop(gt_img, i, j, ph, pw)
                if random.random() < 0.5:
                    input_img = TF.hflip(input_img)
                    gt_img = TF.hflip(gt_img)
            else:
                input_img = TF.center_crop(input_img, [ph, pw])
                gt_img = TF.center_crop(gt_img, [ph, pw])

        return TF.to_tensor(input_img), TF.to_tensor(gt_img)


class ImageFolder_coco(Dataset):
    def __init__(self, root, root_sec, transform=None, split="train"):
        splitdir = Path(root) / split
        splitdir_sec = Path(root_sec) / split
        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')
        if not splitdir_sec.is_dir():
            raise RuntimeError(f'Invalid directory "{root_sec}"')

        
        
        
        self.samples = sorted([f for f in splitdir.iterdir() if f.is_file()])
        self.samples_sec = sorted([f for f in splitdir_sec.iterdir() if f.is_file()])
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        img_sec = Image.open(self.samples_sec[index]).convert("RGB")
        if self.transform:
            return self.transform(img), self.transform(img_sec)
        return img
   

    def __len__(self):
        return len(self.samples)