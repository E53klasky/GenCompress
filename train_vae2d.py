import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler
from tqdm import tqdm
import argparse
import shutil
import time
from models.CDC import keyframe_compressor as compress_modules
from utils import *
from tool.io import save_json


def setup_distributed():
    """
    Initializes torch.distributed if launched via torchrun (RANK / WORLD_SIZE /
    LOCAL_RANK env vars present and WORLD_SIZE > 1). Falls back to plain
    single-GPU mode if launched with a bare `python train_vae2d.py` — so this
    script still runs unchanged for quick local/interactive testing.

    Returns: is_distributed (bool), rank (int), world_size (int), local_rank (int), device
    """
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        # Set the device BEFORE init_process_group, and pass it explicitly via
        # device_id, so NCCL binds this rank to its GPU immediately instead of
        # guessing based on "whatever device happens to be set" — the guess is
        # usually right, but on a wrong guess this can silently hang.
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(
            backend="nccl", rank=rank, world_size=world_size, device_id=device
        )
        return True, rank, world_size, local_rank, device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, 0, device


def is_main_process(rank):
    return rank == 0


def unwrap_model(model):
    """DDP wraps the real model as `model.module`. Use this anywhere you need
    the underlying nn.Module (state_dict saving/loading, etc.)."""
    return model.module if isinstance(model, DDP) else model


def relative_rmse_error_ornl(original, reconstructed, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original = torch.as_tensor(original, dtype=torch.float32, device=device)
    reconstructed = torch.as_tensor(reconstructed, dtype=torch.float32, device=device)

    rmse = torch.sqrt(torch.mean((original - reconstructed) ** 2))
    data_range = torch.max(original) - torch.min(original)

    relative_rmse = torch.where(
        data_range != 0, rmse / data_range, torch.tensor(0.0, device=device)
    )
    return relative_rmse


def train_epoch_vae(
    model, loader, optimizer, scheduler, criterion, loss_beta, device, iteration=0
):
    model.train()
    running_loss1 = torch.zeros(1, device=device)
    running_loss2 = torch.zeros(1, device=device)
    n_samples = 0

    for data_dict in loader:
        inputs = data_dict["input"].to(device, non_blocking=True)
        targets = inputs

        optimizer.zero_grad(set_to_none=True)  # cheaper than zeroing
        results = model(inputs)
        outputs = results["output"]

        loss_mse = criterion(outputs, targets)
        loss_bpp = results["bpp"].mean() * loss_beta
        loss = loss_mse + loss_bpp if loss_beta > 0.0 else loss_mse

        loss.backward()
        optimizer.step()
        scheduler.step()

        # no .item() here — stays a GPU tensor, no sync
        running_loss1 += loss_mse.detach() * inputs.size(0)
        running_loss2 += loss_bpp.detach() * inputs.size(0)
        n_samples += inputs.size(0)
        iteration += 1

    epoch_loss1 = (running_loss1 / n_samples).item()
    epoch_loss2 = (running_loss2 / n_samples).item()

    return epoch_loss1, epoch_loss2, iteration


def test_epoch_vae(model, loader, criterion, device):
    model.eval()
    recons_data = torch.zeros_like(loader.dataset.data_input, device=device)
    bit_count = torch.zeros(1, device=device)

    with torch.no_grad():
        for data_dict in loader:
            inputs = data_dict["input"].to(device, non_blocking=True)
            scale = data_dict["scale"].to(device, non_blocking=True)
            offset = data_dict["offset"].to(device, non_blocking=True)

            results = model(inputs)
            outputs = results["output"] * scale + offset  # stays on GPU
            bit_count += results["frame_bit"].detach().sum()

            idx0, idx1, start_t, end_t = data_dict["index"]
            for i in range(len(inputs)):
                recons_data[idx0[i], idx1[i], start_t[i] : end_t[i]] = outputs[i]

    return recons_data.cpu(), bit_count.item()  # single sync at the very end


class Info:
    def __init__(self, data_name, bpp=32, model_path=None, json_path=None):
        self.json_path = json_path
        self.model_path = model_path

        self.data_name = data_name
        self.bpp = bpp
        self.best_nrmse = 1e10
        self.best_nrmse_cr = 0
        self.best_epoch = -1
        self.all_eval_nrmse = []
        self.all_eval_bpp = []
        self.all_eval_cr = []

    def save_json(self):
        save_json(
            self.json_path,
            {
                self.data_name: {
                    "NRMSE": self.all_eval_nrmse,
                    "best_nrmse": self.best_nrmse,
                    "best_nrmse_cr": self.best_nrmse_cr,
                    "best_index": self.best_epoch,
                    "bpp": self.all_eval_bpp,
                    "cr": self.all_eval_cr,
                }
            },
        )

    def save_last_model(self, model):
        torch.save(model.state_dict(), self.model_path.replace(".pt", "_final.pt"))

    def update(self, model, epoch, nrmse, bpp, dname):
        assert self.data_name == dname
        self.all_eval_nrmse.append(nrmse)
        self.all_eval_bpp.append(bpp)
        self.all_eval_cr.append(self.bpp / bpp)

        if nrmse <= self.best_nrmse:
            torch.save(model.state_dict(), self.model_path)
            self.best_nrmse = nrmse
            self.best_nrmse_cr = self.bpp / bpp
            self.best_epoch = epoch

        self.save_json()


def get_argument():
    parser = argparse.ArgumentParser(
        description="Train a UNet with Channel Attention model."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Per-GPU batch size for training"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./snapshots/E3SM/E3SM_VAE",
        help="Path to save model and results",
    )
    parser.add_argument(
        "--iterations", type=int, default=400, help="Number of epochs for training"
    )

    parser.add_argument(
        "--sr_dim", type=int, default=16, help="Number of epochs for training"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--lr_gamma", type=float, default=0.5, help="Learning rate gamma"
    )

    parser.add_argument("--init_beta", type=float, default=1e-5, help="loss beta")
    parser.add_argument("--end_beta", type=float, default=2e-5, help="loss beta")

    parser.add_argument("--beta_start", type=float, default=0.75, help="loss beta")
    parser.add_argument("--model_dim", type=int, default=16, help="loss beta")
    parser.add_argument("--pretrain", type=str, default="", help="pretrain path")

    # Datatset
    parser.add_argument("--train_set", type=str, default="S3D")
    parser.add_argument("--test_set", type=str, default="E3SM_test")
    parser.add_argument("--config", type=str, default="./configs/config_vae.yaml")

    parser.add_argument(
        "--num_workers",
        type=int,
        default=6,
        help="DataLoader workers PER GPU PROCESS. With multi-GPU/DDP, total worker "
        "processes spawned across all GPUs = num_workers * num_gpus, so reduce this "
        "as you scale up GPUs if you're CPU-limited.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_argument()

    is_distributed, rank, world_size, local_rank, device = setup_distributed()

    save_path = args.save_path

    # Only rank 0 touches the filesystem for setup, to avoid every process
    # racing to create the same directory / copy the same config file.
    if is_main_process(rank):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        shutil.copy(args.config, save_path + "/config_vae.yaml")
    if is_distributed:
        dist.barrier()

    # Input block shapes are fixed across batches, so let cuDNN benchmark and
    # cache the fastest conv algorithm instead of re-searching it each call.
    torch.backends.cudnn.benchmark = True

    # Paths for model and JSON files
    model_path = os.path.join(
        save_path, f"model_bs{args.batch_size}_ep{args.iterations}k.pt"
    )
    json_path = os.path.join(
        save_path, f"model_bs{args.batch_size}_ep{args.iterations}k.json"
    )

    args.iterations = args.iterations * 1000
    if is_distributed:
        # cur_iters counts LOCAL batches per rank, but each rank only sees
        # 1/world_size of the dataset per epoch (DistributedSampler splits it).
        # Without this, hitting the same `cur_iters` target requires world_size
        # times more total data than a single-GPU run with the same
        # --iterations value — i.e. "same setting" silently means "3x more
        # training" once you add GPUs. Dividing here keeps total data
        # processed roughly constant across different GPU counts.
        args.iterations = max(1, args.iterations // world_size)
    if is_main_process(rank):
        save_json(json_path, {"argument": vars(args)})

    train_args = convert_args(args, train=True)

    if is_main_process(rank):
        print(train_args)

    train_datasets = build_dataset(train_args, syn_length=True)
    if is_main_process(rank):
        print("Length for Each dataset", [len(dataset) for dataset in train_datasets])
    merged_dataset = ConcatDataset(train_datasets)

    if is_distributed:
        # DistributedSampler hands each rank a disjoint shard of the dataset each
        # epoch — this replaces shuffle=True (sampler does the shuffling instead).
        train_sampler = DistributedSampler(
            merged_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        train_loader = DataLoader(
            merged_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            merged_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    # Evaluation only ever runs on rank 0 (see training loop below), so only
    # rank 0 needs to build the test datasets/loaders at all.
    test_loaders = []
    if is_main_process(rank):
        test_args = convert_args(args, train=False)
        test_datasets = build_dataset(test_args, syn_length=False)
        test_loaders = [
            DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False, num_workers=3
            )
            for dataset in test_datasets
        ]

    # main(args)
    if args.sr_dim <= 0:
        model = compress_modules.ResnetCompressor(
            dim=args.model_dim,
            dim_mults=[1, 2, 3, 4],
            reverse_dim_mults=[4, 3, 2, 1],
            hyper_dims_mults=[4, 4, 4],
            channels=1,
            out_channels=1,
        )
    else:
        model = compress_modules.CompressorSR(
            dim=args.model_dim,
            dim_mults=[1, 2, 3, 4],
            reverse_dim_mults=[4, 3, 2],
            hyper_dims_mults=[4, 4, 4],
            channels=1,
            out_channels=1,
            sr_dim=args.sr_dim,
        )
        if is_main_process(rank):
            print("loading VAE2D model with SR")

    if args.pretrain != "":
        if is_main_process(rank):
            print("Load pretrain model:", args.pretrain)
        state_dict = torch.load(args.pretrain, map_location="cpu")
        model.load_state_dict(state_dict)

    model = model.to(device)

    if is_distributed:
        # find_unused_parameters=True: this model's entropy bottleneck
        # (FlexiblePrior/hyperprior) has at least one parameter that isn't on
        # the differentiable path every forward pass, which DDP's strict
        # default doesn't allow. This adds a small per-iteration overhead
        # (extra autograd graph traversal to find unused params) but is
        # required here — confirmed by the "Parameter indices which did not
        # receive grad" error without it.
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        if is_main_process(rank):
            print(f"Using DistributedDataParallel across {world_size} GPUs!")
    else:
        if torch.cuda.device_count() > 1:
            print(
                f"{torch.cuda.device_count()} GPUs are visible, but this process "
                "wasn't launched with torchrun, so only 1 GPU will be used. "
                "Launch with: torchrun --nproc_per_node=<N_GPUS> train_vae2d.py ..."
            )
        else:
            print("Using a single GPU!")

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(i / 5 * args.iterations) for i in range(1, 5)],
        gamma=args.lr_gamma,
    )

    loggers = {}
    if is_main_process(rank):
        test_names = [loader.dataset.dataset_name for loader in test_loaders]
        loggers = {name: Info(name, 32, model_path, json_path) for name in test_names}

    cur_iters = 0
    epoch_counter = 0
    is_eval = np.zeros(100, dtype=bool)

    if is_main_process(rank):
        print(
            f"Learning rate milestones: {[int(i/5*args.iterations) for i in range(1, 5)]}"
        )

    #     estimate the remaining time
    start_time = time.time()

    while cur_iters < args.iterations:

        if is_distributed:
            # Reseeds the sampler's shuffling each epoch — without this every
            # epoch would hand out the exact same shard order to each rank.
            train_sampler.set_epoch(epoch_counter)
        epoch_counter += 1

        beta = (
            args.init_beta
            if cur_iters < (args.iterations * args.beta_start)
            else args.end_beta
        )

        mse_loss, bbp_loss, cur_iters = train_epoch_vae(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            beta,
            device,
            cur_iters,
        )
        train_loss = mse_loss + bbp_loss

        eval_index = min(cur_iters // (args.iterations // 100), 99)
        if not is_eval[eval_index]:
            is_eval[eval_index] = True

            # Only rank 0 evaluates and writes checkpoints/logs — model weights
            # are identical across ranks at this point since DDP keeps gradients
            # (and therefore parameters) in sync every step.
            if is_main_process(rank):
                eval_model = unwrap_model(model)
                for test_loader in test_loaders:
                    cur_dataset = test_loader.dataset
                    dname = cur_dataset.dataset_name
                    original_data = cur_dataset.original_data()

                    recons_data, bit_count = test_epoch_vae(
                        eval_model, test_loader, criterion, device
                    )
                    recons_data = cur_dataset.deblocking_hw(recons_data)

                    bpp = float(bit_count / np.prod(recons_data.shape))

                    nrmse = relative_rmse_error_ornl(original_data, recons_data)
                    nrmse = float(nrmse)

                    loggers[dname].update(eval_model, cur_iters, nrmse, bpp, dname)
                    loggers[dname].save_last_model(eval_model)

                    print(
                        dname,
                        f"Progress: {eval_index}/100 ,  Iter {cur_iters}, Train Loss: {train_loss:.6f} ({mse_loss:.6f} + {bbp_loss:.6f})",
                        "NRMSE:",
                        nrmse,
                        f"BPP: {bpp:.6f} CR: {32/bpp:.6f}",
                    )

                    total_time = time.time() - start_time
                    remaining_time = (args.iterations - cur_iters) * (
                        total_time / cur_iters
                    )
                    print(
                        f"Training time: {'%d:%d:%d'%(second_to_time(remaining_time))}/{'%d:%d:%d'%(second_to_time(total_time))}"
                    )

                print()

            # Other ranks wait here while rank 0 evaluates/saves, so everyone
            # re-enters train_epoch_vae together on the next loop iteration —
            # DDP requires every rank to call forward/backward in lockstep.
            if is_distributed:
                dist.barrier()

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()

    if is_main_process(rank):
        print("Training complete.")
