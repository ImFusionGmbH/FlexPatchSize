import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from typing import Optional, Union

import imfusion

imfusion.init()
from imfusion import SharedImageSet


def write_image(im: np.ndarray, path: str) -> None:
    """
    Saves numpy array as a SharedImageSet

    Args:
        im (np.ndarray): Array to be saved, should have the following shape (1, X, Y, Z, C)
        path (str): file name of the saved SharedImageSet (.imf file)
    """
    im = SharedImageSet(im)
    imfusion.io.write([im], path)


def resample_to_img(
    data_np: np.ndarray, template_sis: SharedImageSet, resampling_target: SharedImageSet
) -> SharedImageSet:
    """
    Resample data_np to resampling_target
    This assumes that data_np is at least as large as the template_sis
    If it is larger, extracts a center crop of data_np that matches the template_sis shape

    Args:
        data_np (np.ndarray): data to be resampled
        template_sis (SharedImageSet): template SharedImageSet containing the meta data corresponding to data_np
        resampling_target (SharedImageSet): SharedImageSet the data_np should be resampled to
    """
    templated = template_sis.clone()
    shape = np.array(template_sis).shape
    shape_np = data_np.shape
    for d in range(1, 4):
        if shape[d] == shape_np[d]:
            continue
        l = (shape_np[d] - shape[d]) // 2
        r = (shape_np[d] - shape[d]) - l
        if d == 1:
            data_np = data_np[:, l:-r]
        elif d == 2:
            data_np = data_np[:, :, l:-r]
        elif d == 3:
            data_np = data_np[:, :, :, l:-r]
    templated.assignArray(data_np)
    res = imfusion.executeAlgorithm("Image Resampling", [templated, resampling_target], {"interpolation": "Nearest"})[0]
    return res


def save_case(
    im: torch.Tensor,
    pred: torch.Tensor,
    lbl: torch.Tensor,
    path: str,
    spacing: Optional[np.ndarray] = None,
    matrix: Optional[np.ndarray] = None,
):
    """
    Save training or validation single case

    Args:
        im (torch.tensor): Image to be saved
        pred (torch.tensor): Prediction to be saved
        lbl (torch.tensor): Label to be saved
        path (str): path to the training/validation file
        spacing (np.ndarray): Spacings of the underlying data
        matrix (np.ndarray): Pose matrix
    """
    if len(im.size()) == 5:
        im = im.permute(0, 2, 3, 4, 1).detach().cpu().numpy()
        pred = torch.argmax(pred, dim=1, keepdims=True).permute(0, 2, 3, 4, 1).detach().cpu().numpy()
        lbl = torch.argmax(lbl, dim=1, keepdims=True).permute(0, 2, 3, 4, 1).detach().cpu().numpy()
    elif len(im.size()) == 4:
        im = im.permute(0, 2, 3, 1).detach().cpu().numpy()
        pred = torch.argmax(pred, dim=1, keepdims=True).permute(0, 2, 3, 1).detach().cpu().numpy()
        lbl = torch.argmax(lbl, dim=1, keepdims=True).permute(0, 2, 3, 1).detach().cpu().numpy()

    im = SharedImageSet(im)
    # im.modality = imfusion.Data.Modality.MRI

    lbl = SharedImageSet(lbl.astype(np.uint8))
    lbl.modality = imfusion.Data.Modality.LABEL
    lbl.name = "GT"

    pred = SharedImageSet(pred.astype(np.uint8))
    pred.modality = imfusion.Data.Modality.LABEL
    pred.name = "Pred"

    if spacing is not None:
        im[0].spacing = spacing
        lbl[0].spacing = spacing
        pred[0].spacing = spacing

    if matrix is not None:
        im.set_matrix(matrix)
        lbl.set_matrix(matrix)
        pred.set_matrix(matrix)

    imfusion.io.write([im, lbl, pred], path)


def moving_average(data: Union[list, np.ndarray], k: int) -> Union[list, np.ndarray]:
    """
    computes a moving average of the data with a rolling window of size k

    Args:
        data (np.ndarray): data to be running averaged
        k (int): size of the rolling window
    """
    if len(data) < k + 1:
        return data
    else:
        return [np.mean(data[i : i + k]) for i in range(len(data) - k)]


def block_average(data: Union[list, np.ndarray], k: int) -> Union[list, np.ndarray]:
    """
    computes a block average of the data with window of size k

    Args:
        data (np.ndarray): data to be block averaged
        k (int): size of the window
    """
    if len(data) < k:
        return data
    else:
        return [np.mean(data[i * k : (i + 1) * k]) for i in range(len(data) // k)] + [np.mean(data[-k:])]


def plot_loss(loss_dict: dict, path: str, ma: int = 100) -> None:
    """
    Plot training log

    Args:
    loss_dict (dict): log object in the form a dictionary as follow
                        {
                            phase: {
                                loss_name : [list_values]
                            }
                        }
    path (str): path to image file for the plot
    ma (int): moving average window size
    """
    # Only plot varying losses
    n_losses_train = len(
        [k for k in loss_dict.get("Train", {}).keys() if np.std(loss_dict.get("Train", {}).get(k, [0])) > 0.00001]
    )
    n_losses_val = len(
        [k for k in loss_dict.get("Val", {}).keys() if np.std(loss_dict.get("Val", {}).get(k, [0])) > 0.00001]
    )
    n_losses = max(n_losses_train, n_losses_val)
    fig, axes = plt.subplots(2, n_losses, figsize=(15 * n_losses, 20))

    for i, phase in enumerate(["Train", "Val"]):
        idx = 0
        c = "blue" if phase == "Train" else "orange"
        for loss_name in loss_dict.get(phase, {}):
            if np.std(loss_dict[phase][loss_name]) <= 0.00001:
                continue
            data = (
                loss_dict[phase][loss_name] if loss_name == "Raw" else moving_average(loss_dict[phase][loss_name], ma)
            )
            axes[i][idx].plot(np.arange(len(data)), data, color=c)
            axes[i][idx].set_title(phase + " " + loss_name)
            idx += 1

    fig.savefig(path, bbox_inches="tight")
    plt.clf()


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Dice loss

    Args:
        pred (torch.Tensor): prediction tensor
        target (torch.Tensor): target tensor
    """
    axes = list(range(2, len(pred.size())))
    return 1 - (2 * (pred * target).sum(dim=axes) + 1e-8) / (pred.sum(dim=axes) + target.sum(dim=axes) + 1e-8)


def cross_entropy_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Cross Entropy loss

    Args:
        pred (torch.Tensor): prediction tensor
        target (torch.Tensor): target tensor
    """
    return F.cross_entropy(pred, target)


def log_dice_loss(log: dict, dice: torch.Tensor, phase: str, verbose: bool = False, prefix: str = "") -> dict:
    """
    Log dice loss in log

    Args:
        log (dict): log
        dice (torch.Tensor): Dice loss tensor
        phase (str): phase which the dice tensor originated from
        verbose (bool): if True, prints the mean Dice score per class
        prefix (str): prefix for the log entry (prepended to 'dice class_num')
    """
    n_labels = dice.size(1)
    dice_np = dice.detach().cpu().numpy()
    for n in range(n_labels):
        log[phase][prefix + "Dice " + str(n)] = log[phase].get(prefix + "Dice " + str(n), []) + list(dice_np[:, n])

    if verbose:
        print(", ".join([f"Dice {i}: {dice_np[:, i].mean()}" for i in range(n_labels)]))
    return log


def makedir(dir: str) -> None:
    """
    Create a directory if it does not exist

    Args:
        dir (str): directory to be created
    """
    if not os.path.isdir(dir):
        os.makedirs(dir)


def split_file(file_name: str, p_train: float, file_train: str, file_val: str) -> None:
    """
    Splits a data list file into training and validation data list files

    Args:
        file_name (str): data list file to be splitted
        p_train (float): proportion of the data list to be used for training
        file_train (str): path to training data list file
        file_val (str): path to validation data list file
    """
    with open(file_name, "r") as f:
        file_all = f.readlines()

    train_list = [file_all[0]]
    val_list = [file_all[0]]

    perm = np.random.permutation(len(file_all) - 1)
    n_train = int((len(file_all) - 1) * p_train)

    for i, line in enumerate(file_all[1:]):
        if i in perm[:n_train]:
            train_list.append(line)
        else:
            val_list.append(line)

    train_list = "\n".join(train_list)
    val_list = "\n".join(val_list)

    with open(file_train, "w") as f:
        f.write(train_list)

    with open(file_val, "w") as f:
        f.write(val_list)


def convert_to_uint(im: SharedImageSet) -> SharedImageSet:
    """
    Changes image data type to unsigned char

    Args:
        im (SharedImageSet): SharedImageSet whose data type needs to be changed
    """
    im_np = np.array(im)
    new_im = SharedImageSet(im_np.astype(np.uint8))
    new_im[0].spacing = im[0].spacing
    new_im[0].matrix = im[0].matrix
    return new_im
