from pathlib import Path
from typing import Dict, Optional

import torch


def _validate_dimensions(state_dim: int, node_pressure_dim: int, equipment_dims: int, theta_dim: int) -> None:
    if state_dim <= 0 or node_pressure_dim <= 0 or equipment_dims <= 0 or theta_dim <= 0:
        raise ValueError("state_dim, node_pressure_dim, equipment_dims, theta_dim must be positive integers")
    if state_dim < equipment_dims:
        raise ValueError("state_dim must be greater than or equal to equipment_dims to build H")


def build_pipeline_matrices(
    data_dir: str,
    state_dim: int,
    node_pressure_dim: int,
    equipment_dims: int,
    theta_dim: int,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Construct placeholder pipeline matrices for PIRN.

    Args:
        data_dir: Base data directory (kept for future extensions that read static files).
        state_dim: Internal state dimension.
        node_pressure_dim: Node pressure dimension.
        equipment_dims: Equipment output dimension.
        theta_dim: Trainable parameter dimension.
        device: Target torch device.

    Returns:
        Dictionary of torch tensors required by the PIRN model.
    """
    _ = Path(data_dir)
    if device is None:
        device = torch.device("cpu")

    _validate_dimensions(state_dim, node_pressure_dim, equipment_dims, theta_dim)

    total_dim = state_dim + node_pressure_dim

    K_base = torch.eye(total_dim, dtype=torch.float32, device=device)
    S = torch.eye(state_dim, dtype=torch.float32, device=device)

    H = torch.zeros(equipment_dims, state_dim, dtype=torch.float32, device=device)
    equipment_slice = H[:equipment_dims, :equipment_dims]
    equipment_slice.copy_(torch.eye(equipment_slice.shape[0], dtype=torch.float32, device=device))

    theta0 = torch.zeros(theta_dim, dtype=torch.float32, device=device)

    M_mat = torch.zeros(total_dim, theta_dim, dtype=torch.float32, device=device)
    N_mat = torch.zeros(total_dim, theta_dim, dtype=torch.float32, device=device)

    mask_state = torch.zeros(total_dim, dtype=torch.bool, device=device)
    mask_state[:state_dim] = True
    mask_node = torch.zeros(total_dim, dtype=torch.bool, device=device)
    mask_node[state_dim:] = True

    return {
        "K_base": K_base,
        "S": S,
        "H": H,
        "theta0": theta0,
        "M_mat": M_mat,
        "N_mat": N_mat,
        "mask_state": mask_state,
        "mask_node": mask_node,
    }
