from __future__ import annotations

import torch
from torch.utils.data import Dataset
from ocrap.teacher.dataset_writer import read_dataset


class RecoveryDataset(Dataset):
    def __init__(self, path: str):
        self.arrays, self.metadata = read_dataset(path)
        self.N = self.arrays["bev"].shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        a = self.arrays
        keys = ["bev", "ego_info", "route_command", "actions_states", "actions_controls", "actions_params", "token_states_ref", "token_controls_ref", "token_params", "token_anchor", "token_hard_shell", "options_states_ref", "options_controls_ref", "options_params", "action_mask", "option_mask", "mode_probs", "g_star", "y_star", "h_star", "k_star", "u_star", "c_rule_star", "spec_margin_star", "spec_id_star", "margin_option", "obs_class", "obs_equiv", "beta_star", "witness_oc", "Y_oc", "R_star", "P_star", "G_star", "C_star", "U_star", "H_star", "K_star", "witness"]
        out = {}
        for k in keys:
            if k in a:
                out[k] = torch.as_tensor(a[k][idx])
        # Map old option names to OC-RAP names for compatibility.
        if "token_states_ref" not in out and "options_states_ref" in out:
            out["token_states_ref"] = out["options_states_ref"]
        if "token_controls_ref" not in out and "options_controls_ref" in out:
            out["token_controls_ref"] = out["options_controls_ref"]
        if "token_params" not in out and "options_params" in out:
            out["token_params"] = out["options_params"]
        if "token_anchor" not in out and "options_params" in out:
            # Backward-compatible fallback for old datasets: use terminal target
            # x/y from option params and zero heading.  New datasets should carry
            # explicit token_anchor from recovery_options.options_to_tensors().
            p = out["options_params"].float()
            out["token_anchor"] = torch.stack([p[..., 4], p[..., 5], torch.zeros_like(p[..., 4])], dim=-1)
        if "token_hard_shell" not in out and "option_mask" in out:
            valid = out["option_mask"].float()
            out["token_hard_shell"] = torch.stack([valid, valid, valid, valid * 2.0 - 1.0], dim=-1)
        return out
