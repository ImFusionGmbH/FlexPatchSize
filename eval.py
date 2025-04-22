import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
import pickle
import shutil

from utils import *
from data import *
from models import *


def main(config: dict, config_path: str) -> None:
    """
    Main function to run the inference
    Args:
        config (dict): configuration dictionary
        config_path (str): path to the configuration file
    """
    # Model config
    model_config = config["model"]
    model_type = config["model_type"]

    # Dataset config
    data_file = config["dataset"]["data_file"]
    spacing = config["dataset"]["spacing"]
    n_labels = config["dataset"]["n_labels"]
    bs = config["dataset"]["batch_size"]
    flip_image_content = config["dataset"].get("flip_image_content", False)
    flip_axes = config["dataset"].get("flip_axes", None)

    # Dictionary containing the patch sizes to be used (e.g. {"0": [32, 32, 32], "1": [64, 128, 72], ...})
    ps_dict_path = config.get("patch_size_dict", "")

    # Model
    model_config.update({"out_c": n_labels + 1})
    if model_type == "HyperPatch":
        model = HyperUnet(**model_config).cuda()
    elif model_type == "UNet":
        model = UNet(**model_config).cuda()
    else:
        raise ValueError(f"Unknown model type {model_type}")

    if config["checkpoint"]:
        checkpoint = torch.load(config["checkpoint"])
        model.load_state_dict(checkpoint["model"])

    model.eval()

    # Logs
    path_log = Path(config["output_path"]) / "inference"
    path_log.mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, path_log / "config_inference.yaml")

    if ps_dict_path:
        with open(ps_dict_path, "rb") as f:
            ps_dict = pickle.load(f)
    # Isotropic patch sizes
    else:
        min_ps = 32
        max_ps = 264
        step = 2 ** model_config["n_down"]
        ps_dict = {i: [ps] * 3 for i, ps in enumerate(range(min_ps, max_ps, step))}

    # Run inference per each patch size
    for _, ps in ps_dict.items():
        ps_str = "x".join(map(str, ps))
        results_dir = path_log / f"patch_size_{ps_str}"

        if (results_dir / "dice.pkl").is_file():
            continue
        results_dir.mkdir(parents=True, exist_ok=True)

        dataset = BasicDataset(
            data_file,
            spacing=spacing,
            n_labels=n_labels,
            flip_image_content=flip_image_content,
            flip_axes=flip_axes,
        )
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)

        ps = np.array(ps)
        ps = ps[None, ...]  # add batch dimension
        ps = np.repeat(ps, repeats=bs, axis=0)
        ps = torch.Tensor(ps).cuda()

        inference(dataloader, model, ps, results_dir, device="cuda")


def inference(
    dataloader: DataLoader,
    model: nn.Module,
    ps: torch.Tensor,
    save_dir: Path,
    save_cases: bool = False,
    device: str = "cuda",
) -> None:
    """
    Run inference on the whole image

    Args:
        dataloader (DataLoader): dataloader
        model (nn.Module): model
        ps (torch.Tensor): patch size as conditioning variable
        save_dir (Path): directory to save the results
        save_cases (bool): whether to save the cases or not
        device (str): device to run the inference on
    """

    dice_dict = {}
    with torch.no_grad():
        model.eval()
        for it, (batch) in tqdm(enumerate(dataloader)):
            # Load data
            im = batch["im"].to(device)
            lbl = batch["lbl"].to(device)
            identifiers = batch["identifier"]

            pred_log = patch_based_inference(im, model, ps)
            pred_prob = F.softmax(pred_log, dim=1)

            dice = 1 - dice_loss(pred_prob, lbl)[:, 1:]
            dice_dict.update(((case_id, d) for case_id, d in zip(identifiers, dice)))

            if save_cases:
                for case_id, im_i, lbl_i, pred_i in zip(identifiers, im, lbl, pred_prob):
                    # Recover batch dimension
                    im_i = im_i[None, ...]
                    lbl_i = lbl_i[None, ...]
                    pred_i = pred_i[None, ...]

                    save_case(im, pred_prob, lbl, str(save_dir / f"{case_id}.imf"))

    with open(save_dir / "dice.pkl", "wb") as f:
        pickle.dump(dice_dict, f)

    return


def make_size_divisible(im: torch.Tensor, div: int) -> tuple[torch.Tensor, slice, slice, slice]:
    """
    Pads an image to make its dimensions divible by div
    Returns the padded image and the slices to retrieve the original image content

    Args:
        im (torch.Tensor): image tensor
        div (int): desired dividor of the image dimensions
    """
    B, C, X, Y, Z = im.size()
    r_z = Z % div
    pad_z_m = (div - r_z) // 2 if r_z != 0 else 0
    pad_z_p = (div - r_z) - pad_z_m if r_z != 0 else 0
    r_y = Y % div
    pad_y_m = (div - r_y) // 2 if r_y != 0 else 0
    pad_y_p = (div - r_y) - pad_y_m if r_y != 0 else 0
    r_x = X % div
    pad_x_m = (div - r_x) // 2 if r_x != 0 else 0
    pad_x_p = (div - r_x) - pad_x_m if r_x != 0 else 0
    im = F.pad(im, (pad_z_m, pad_z_p, pad_y_m, pad_y_p, pad_x_m, pad_x_p), mode="replicate")
    return (
        im,
        slice(pad_x_m, -pad_x_p or X),
        slice(pad_y_m, -pad_y_p or Y),
        slice(pad_z_m, -pad_z_p or Z),
    )


def patch_based_inference(im: torch.Tensor, model: nn.Module, ps: torch.Tensor) -> torch.Tensor:
    """
    Perform patch-based inference on an image

    Args:
        im (torch.Tensor): image tensor
        model (nn.Module): model
        ps (torch.Tensor): patch size as conditioning variable, potentially normalised
    """
    patch_size = ps[0].cpu().numpy().astype(np.uint32).tolist()

    im, sx, sy, sz = make_size_divisible(im, 8)

    half_patch_size = [(patch_size[0] // 2), (patch_size[1] // 2), (patch_size[2] // 2)]
    B, C, X, Y, Z = im.size()

    # Determine the number of steps the window will move in each dimension
    if patch_size[0] > X:
        n_x = 1
    else:
        n_x = (X - half_patch_size[0]) // half_patch_size[0]
        if (X - half_patch_size[0]) % half_patch_size[0] != 0:
            n_x += 1

    if patch_size[1] > Y:
        n_y = 1
    else:
        n_y = (Y - half_patch_size[1]) // half_patch_size[1]
        if Y % half_patch_size[1] != 0:
            n_y += 1

    if patch_size[2] > Z:
        n_z = 1
    else:
        n_z = (Z - half_patch_size[2]) // half_patch_size[2]
        if Z % half_patch_size[2] != 0:
            n_z += 1

    res = torch.zeros((B, model.out_c, X, Y, Z))

    # Patch weights
    patch_weight = torch.exp(
        -(
            (
                torch.stack(
                    torch.meshgrid(
                        torch.linspace(-1, 1, min(patch_size[0], X)),
                        torch.linspace(-1, 1, min(patch_size[1], Y)),
                        torch.linspace(-1, 1, min(patch_size[2], Z)),
                    ),
                    dim=-1,
                )
            )
            ** 2
        ).sum(-1)
        / (2.0)
    )[None, None, ...].cuda()

    with torch.no_grad():
        for x in range(n_x):
            x_start = x * half_patch_size[0] if (x + 2) * half_patch_size[0] <= X else -patch_size[0]
            x_end = (x + 2) * half_patch_size[0] if (x + 2) * half_patch_size[0] <= X else X
            slice_x = slice(x_start, x_end)
            if patch_size[0] > X:
                slice_x = slice(0, X)
            for y in range(n_y):
                y_start = y * half_patch_size[1] if (y + 2) * half_patch_size[1] <= Y else -patch_size[1]
                y_end = (y + 2) * half_patch_size[1] if (y + 2) * half_patch_size[1] <= Y else Y
                slice_y = slice(y_start, y_end)
                if patch_size[1] > Y:
                    slice_y = slice(0, Y)
                for z in range(n_z):
                    z_start = z * half_patch_size[2] if (z + 2) * half_patch_size[2] <= Z else -patch_size[2]
                    z_end = (z + 2) * half_patch_size[2] if (z + 2) * half_patch_size[2] <= Z else Z
                    slice_z = slice(z_start, z_end)
                    if patch_size[2] > Z:
                        slice_z = slice(0, Z)
                    pred = model(im[:, :, slice_x, slice_y, slice_z].cuda(), ps) * patch_weight
                    res[:, :, slice_x, slice_y, slice_z] += pred.cpu()

    return res[:, :, sx, sy, sz].to(device="cuda")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="config_inference.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config, args.config)
