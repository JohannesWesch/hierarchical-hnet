"""
Distributed training utilities for H-Net.

This module provides utilities for:
- Setting up distributed training
- Data parallel and distributed data parallel
- Gradient synchronization
- Checkpointing in distributed settings
"""

import os
import warnings
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Setup distributed training environment.

    Args:
        backend: Distributed backend (nccl, gloo, mpi)
        init_method: URL for process group initialization

    Returns:
        Dictionary with rank, world_size, local_rank
    """
    # Check if we're in SLURM environment
    if "SLURM_PROCID" in os.environ:
        # SLURM environment
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])

        # Set CUDA device
        torch.cuda.set_device(local_rank)

        # Initialize process group
        if init_method is None:
            # Use SLURM environment for initialization
            master_addr = os.environ.get("SLURM_LAUNCH_NODE_IPADDR", "localhost")
            master_port = os.environ.get("SLURM_LAUNCH_NODE_PORT", "12355")
            init_method = f"tcp://{master_addr}:{master_port}"

        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
    else:
        # Local environment (for testing)
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if world_size > 1:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend=backend,
                init_method=init_method or "env://",
                rank=rank,
                world_size=world_size,
            )

    return {
        "rank": rank,
        "world_size": world_size,
        "local_rank": local_rank,
    }


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_local_rank() -> int:
    """Get local rank (rank within node)."""
    if "SLURM_LOCALID" in os.environ:
        return int(os.environ["SLURM_LOCALID"])
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    return get_rank() == 0


def barrier():
    """Synchronization barrier across all processes."""
    if dist.is_initialized():
        dist.barrier()


def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """
    All-reduce operation on tensor.

    Args:
        tensor: Input tensor
        op: Reduction operation

    Returns:
        Reduced tensor
    """
    if dist.is_initialized():
        dist.all_reduce(tensor, op=op)
    return tensor


def all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-gather operation on tensor.

    Args:
        tensor: Input tensor

    Returns:
        Gathered tensor
    """
    if dist.is_initialized():
        world_size = get_world_size()
        tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor)
        return torch.stack(tensor_list, dim=0)
    return tensor.unsqueeze(0)


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Broadcast tensor from source rank to all ranks.

    Args:
        tensor: Tensor to broadcast
        src: Source rank

    Returns:
        Broadcasted tensor
    """
    if dist.is_initialized():
        dist.broadcast(tensor, src=src)
    return tensor


def setup_ddp_model(
    model: torch.nn.Module,
    device_ids: Optional[list] = None,
    find_unused_parameters: bool = False,
) -> DDP:
    """
    Wrap model with DistributedDataParallel.

    Args:
        model: Model to wrap
        device_ids: List of device IDs
        find_unused_parameters: Whether to find unused parameters

    Returns:
        DDP-wrapped model
    """
    if not dist.is_initialized():
        warnings.warn("Distributed not initialized, returning unwrapped model", stacklevel=2)
        return model

    if device_ids is None:
        device_ids = [get_local_rank()]

    return DDP(
        model,
        device_ids=device_ids,
        output_device=device_ids[0],
        find_unused_parameters=find_unused_parameters,
    )


def reduce_dict(
    input_dict: Dict[str, torch.Tensor],
    average: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Reduce dictionary of tensors across all processes.

    Args:
        input_dict: Dictionary of tensors to reduce
        average: Whether to average (True) or sum (False)

    Returns:
        Reduced dictionary
    """
    if not dist.is_initialized():
        return input_dict

    reduced_dict = {}
    for key, tensor in input_dict.items():
        if torch.is_tensor(tensor):
            reduced_tensor = tensor.clone()
            dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
            if average:
                reduced_tensor /= get_world_size()
            reduced_dict[key] = reduced_tensor
        else:
            reduced_dict[key] = tensor

    return reduced_dict


def save_checkpoint_distributed(
    checkpoint: Dict[str, Any],
    filepath: str,
    is_main_process: bool,
):
    """
    Save checkpoint in distributed setting.

    Only saves from main process to avoid conflicts.

    Args:
        checkpoint: Checkpoint dictionary
        filepath: Path to save checkpoint
        is_main_process: Whether current process is main
    """
    if is_main_process:
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)


def load_checkpoint_distributed(
    filepath: str,
    map_location: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load checkpoint in distributed setting.

    Args:
        filepath: Path to checkpoint
        map_location: Device to load to

    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filepath, map_location=map_location)

    # Broadcast checkpoint metadata from rank 0
    if dist.is_initialized():
        # For now, just load on rank 0 and broadcast key info
        # In practice, you might want to load different parts on different ranks
        pass

    return checkpoint


class DistributedSampler:
    """
    Distributed sampler for training data.

    Ensures each process gets different data samples.
    """

    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        """
        Initialize distributed sampler.

        Args:
            dataset: Dataset to sample from
            num_replicas: Number of processes
            rank: Current process rank
            shuffle: Whether to shuffle data
            seed: Random seed
        """
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        if hasattr(dataset, "__len__"):
            self.num_samples = len(dataset)
        else:
            self.num_samples = None

    def __iter__(self):
        """Iterate over indices."""
        if self.num_samples is None:
            # For iterable datasets, we can't determine length
            # This is handled by the DistributedIterableDatasetWrapper
            yield from iter(range(float("inf")))
            return

        # Generate indices
        indices = list(range(self.num_samples))

        if self.shuffle:
            # Use epoch for shuffling
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = indices[torch.randperm(len(indices), generator=g).tolist()]

        # Subsample for this rank
        per_replica = len(indices) // self.num_replicas
        remainder = len(indices) % self.num_replicas

        start_idx = self.rank * per_replica + min(self.rank, remainder)
        end_idx = start_idx + per_replica + (1 if self.rank < remainder else 0)

        yield from indices[start_idx:end_idx]

    def __len__(self) -> int:
        """Get number of samples."""
        if self.num_samples is None:
            return 0

        per_replica = self.num_samples // self.num_replicas
        remainder = self.num_samples % self.num_replicas

        return per_replica + (1 if self.rank < remainder else 0)

    def set_epoch(self, epoch: int):
        """Set epoch for shuffling."""
        self.epoch = epoch
