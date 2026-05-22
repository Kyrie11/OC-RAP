from __future__ import annotations

import torch
from torch.utils.data import Dataset
from recap.teacher.dataset_writer import read_dataset


class RecoveryDataset(Dataset):
    def __init__(self, path: str):
        self.arrays, self.metadata = read_dataset(path)
        self.N = self.arrays["bev"].shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        a = self.arrays
        keys = ["bev", "ego_info", "route_command", "actions_states", "options_states_ref", "action_mask", "option_mask", "P_star", "G_star", "C_star", "U_star", "H_star", "K_star", "mode_probs", "witness"]
        out = {}
        for k in keys:
            if k in a:
                out[k] = torch.as_tensor(a[k][idx])
        return out
