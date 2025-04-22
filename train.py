import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from typing import Optional
import yaml
import argparse
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import shutil

from utils import *
from data import *
from models import *


def initialize_weights(m: nn.Module) -> None:
    """
    Initialize weights of the hypernetwork
    Args:
        m (nn.Module): model module
    """
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.normal_(m.weight, 0, 0.025)
        torch.nn.init.normal_(m.bias, 0, 0.005)


def main(config: dict, config_path: str) -> None:
    """
    Main function to run the training loop

    Args:
        config (dict): configuration dictionary
        config_path (str): path to the configuration file
    """
    # ==== Parse config ===

    # Model config
    model_config = config["model"]
    model_type = config["model_type"]

    # Dataset config
    data_file_train = config["dataset"]["data_file_train"]
    data_file_val = config["dataset"]["data_file_val"]
    spacing = config["dataset"]["spacing"]
    n_labels = config["dataset"]["n_labels"]
    patch_size_range = config["dataset"]["patch_size_range"]
    use_fixed_patch_size = config["dataset"]["use_fixed_patch_size"]
    fixed_patch_size = config["dataset"]["fixed_patch_size"]
    only_divisible_patch_sizes = config["dataset"]["only_divisible_patch_sizes"]

    # Training config
    epochs = config["training"]["epochs"]
    repeat = config["training"]["repeat"]
    accumulation_steps = config["training"]["accumulation_steps"]
    lr = config["training"].get("lr", 0.001)
    bs = config["training"].get("batch_size", 1)

    # ==== Dataset instantiation ====
    dataset_train = PatchDataset(
        data_file_train,
        n_downsample=model_config["n_down"],
        spacing=spacing,
        n_labels=n_labels,
        patch_size_range=patch_size_range,
        use_fixed_patch_size=use_fixed_patch_size,
        fixed_patch_size=fixed_patch_size,
        only_divisible_patch_sizes=only_divisible_patch_sizes,
    )
    dataset_val = PatchDataset(
        data_file_val,
        n_downsample=model_config["n_down"],
        spacing=spacing,
        n_labels=n_labels,
        patch_size_range=patch_size_range,
        use_fixed_patch_size=use_fixed_patch_size,
        fixed_patch_size=fixed_patch_size,
        only_divisible_patch_sizes=only_divisible_patch_sizes,
    )

    train_dataloader = DataLoader(dataset_train, batch_size=bs, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=bs, shuffle=False)

    # ==== Model ====
    model_config.update({"out_c": n_labels + 1})
    if model_type == "HyperPatch":
        model = HyperUnet(**model_config).cuda()
    elif model_type == "UNet":
        model = UNet(**model_config).cuda()
    else:
        raise ValueError(f"Unknown model type {model_type}")

    # Initialize weights with small std to counteract the raw big patch values
    if isinstance(model, HyperUnet):
        model.apply(initialize_weights)

    # ==== Optimizer ====
    opt = Adam(model.parameters(), lr=lr)

    if config["checkpoint"]:
        checkpoint = torch.load(config["checkpoint"])
        model.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["optimizer"])

    # ==== Logs ====
    path_log = Path(config["output_path"])
    path_log.mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, path_log / "config.yaml")

    # ==== Launch training ====
    train(
        train_dataloader,
        val_dataloader,
        model,
        opt,
        epochs,
        path_log,
        device="cuda",
        repeat=repeat,
        accumulation_steps=accumulation_steps,
    )


def train(
    dataloader_train: DataLoader,
    dataloader_val: DataLoader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    n_epoch: int,
    save_dir: Path,
    device: str = "cuda",
    repeat: int = 1,
    accumulation_steps: int = 1,
) -> None:
    """
    Run the training loop for a model

    Args:
        dataloader_train (DataLoader): train dataloader
        dataloader_val (DataLoader): validation dataloader
        model (nn.Module): model to train
        optimizer (torch.optim.Optimizer): torch optimizer optimizing the model
        n_epoch (int): number of training epochs to run
        save_dir (str): Path to directory where results should be saved
        device (str): device on which to run the training (should be 'cuda')
        repeat (int): Number of training epochs between validations
        accumulation_steps (int): Number of batches to accumulate gradients over
    """

    training_dir = save_dir / "training_results"
    training_dir.mkdir(exist_ok=True)

    # statistics
    log = {"Train": defaultdict(list), "Val": defaultdict(list)}
    log["Train"]["EMA totloss"] = [0]
    log["Val"]["EMA totloss"] = [0]

    log = validation(dataloader_val, model, save_dir, log, save=True, epoch=0)

    torch.cuda.empty_cache()

    best_val_loss = np.inf

    for e in range(n_epoch):
        print(f"Starting epoch {e}...\n")
        optimizer.zero_grad()
        for it, (batch) in tqdm(enumerate(dataloader_train)):
            # load data
            im = batch["im"].to(device)
            lbl = batch["lbl"].to(device)
            ps = batch["ps"].to(device)

            pred_log = model(im, ps)
            pred_prob = F.softmax(pred_log, dim=1)

            # loss
            dice_err = dice_loss(pred_prob, lbl)
            ce_err = cross_entropy_loss(pred_log, lbl)

            loss = dice_err.mean() + ce_err

            # logs
            dice_err_no_bg = dice_err[:, 1:]  # skip background in logs
            log = log_dice_loss(log, dice_err_no_bg, "Train", verbose=False, prefix="")
            log["Train"]["CE"].append(ce_err.item())
            log["Train"]["LogDice"].append(np.log(dice_err.mean().item() + 1e-8))
            log["Train"]["LogCE"].append(np.log(ce_err.item() + 1e-8))
            log["Train"]["EMA totloss"] += [0.1 * loss.item() + 0.9 * log["Train"]["EMA totloss"][-1]]

            # Gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            if (it + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # Save firts 8 cases from the current epoch
            if it < 8:
                save_case(im, pred_prob, lbl, str(training_dir / f"case{it}.imf"))

        # Handle any remaining gradients at the end of the epoch
        if (it + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()

        if e % repeat == 0:
            log = validation(dataloader_val, model, save_dir, log, save=True, epoch=e)

            torch.cuda.empty_cache()

            plot_loss(log, os.path.join(save_dir, "loss.png"))

            # Save the latest model
            torch.save(model.state_dict(), training_dir / "model_w.pt")
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                training_dir / "checkpoint.pt",
            )

            # Save the model for the best val loss
            if log["Val"]["Raw"][-1] < best_val_loss:
                best_val_loss = log["Val"]["Raw"][-1]

                # Remove previous saved model
                for c in training_dir.glob("*.pt"):
                    # Remove only best last model
                    if "epoch" in str(c):
                        c.unlink()

                torch.save(model.state_dict(), training_dir / f"model_w_epoch{e}.pt")
                torch.save(
                    {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                    training_dir / f"checkpoint_epoch{e}.pt",
                )

    torch.save(model.state_dict(), save_dir / "final_model_w.pt")

    return


def validation(
    dataloader: DataLoader,
    model: nn.Module,
    save_dir: str,
    log: dict,
    save: bool = False,
    device: str = "cuda",
    epoch: Optional[int] = None,
):
    """
    Run validation for a model

    Args:
        dataloader (DataLoader): validation dataloader
        model (nn.Module): model
        save_dir (str): Path where to save results
        log (dict): log dictionary
        save (bool): if True, the best and worst cases are saved
        device (str): device on which to run the validation (should be 'cuda')
        epoch (int): if provided, add the epoch number to the validation file name
    """
    print("Running evaluation...\n")

    best_case = [np.inf, None, None, None]  # [loss, im, lbl, pred]
    worst_case = [-np.inf, None, None, None]
    loss_list = []

    with torch.no_grad():
        model.eval()
        for it, batch in enumerate(dataloader):

            # load data
            im = batch["im"].to(device)
            lbl = batch["lbl"].to(device)
            ps = batch["ps"].to(device)

            pred_log = model(im, ps)
            pred_prob = F.softmax(pred_log, dim=1)

            # loss
            dice_err = dice_loss(pred_prob, lbl)
            ce_err = cross_entropy_loss(pred_log, lbl)
            loss = ce_err.mean() + dice_err.mean()
            loss = float(loss.detach().cpu().numpy())
            loss_list.append(loss)

            # logs
            dice_err_no_bg = dice_err[:, 1:]  # skip background in logs
            log = log_dice_loss(log, dice_err_no_bg, "Val", verbose=False, prefix="")
            log["Val"]["CE"].append(ce_err.item())
            log["Val"]["LogDice"].append(np.log(dice_err.mean().item() + 1e-8))
            log["Val"]["LogCE"].append(np.log(ce_err.item() + 1e-8))
            log["Val"]["EMA totloss"] += [0.1 * loss + 0.9 * log["Val"]["EMA totloss"][-1]]

            bs = im.shape[0]
            idx = np.random.randint(0, bs)
            if loss < best_case[0]:
                best_case = [
                    loss,
                    torch.unsqueeze(im[idx], 0),
                    torch.unsqueeze(lbl[idx], 0),
                    torch.unsqueeze(pred_log[idx], 0),
                ]
            if loss > worst_case[0]:
                worst_case = [
                    loss,
                    torch.unsqueeze(im[idx], 0),
                    torch.unsqueeze(lbl[idx], 0),
                    torch.unsqueeze(pred_log[idx], 0),
                ]

        # Save validation loss value per epoch (without background)
        log["Val"]["Raw"] += [np.mean(loss_list)]

        if save:
            validation_dir = save_dir / "validation_results"
            validation_dir.mkdir(exist_ok=True)

            # Save best case
            path_save_case = (
                validation_dir / f"epoch{epoch}_best.imf"
                if isinstance(epoch, int)
                else validation_dir / "validation_best.imf"
            )

            im, lbl, pred_prob = best_case[1:]
            save_case(im, pred_prob, lbl, str(path_save_case))

            # Save worst case
            path_save_case = (
                validation_dir / f"epoch{epoch}_worst.imf"
                if isinstance(epoch, int)
                else validation_dir / "validation_worst.imf"
            )
            im, lbl, pred_prob = worst_case[1:]
            save_case(im, pred_prob, lbl, str(path_save_case))

    model.train()
    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config, args.config)
