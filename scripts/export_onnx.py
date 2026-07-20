"""Standalone script to export a trained RSL-RL policy to ONNX format.

Usage:
    python scripts/export_onnx.py --checkpoint logs/rsl_rl/unitree_g1_29dof_velocity/2026-05-15_11-27-27/model_4999.pt
"""

import argparse
import os

import torch
import torch.nn as nn


class ActorPolicy(nn.Module):
    """Reconstructed actor-only policy for ONNX export (deterministic mean action)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int], activation: str = "elu"):
        super().__init__()

        act_fn = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "selu": nn.SELU}[activation]

        layers = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), act_fn()]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.mlp = nn.Sequential(*layers)

        # GaussianDistribution keeps a learnable std — needed to match state_dict keys
        self.distribution = nn.ParameterDict({"std_param": nn.Parameter(torch.zeros(action_dim))})

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model_XXXX.pt checkpoint")
    parser.add_argument("--output", default=None, help="Output .onnx path (default: <checkpoint_dir>/exported/policy.onnx)")
    parser.add_argument("--obs-dim", type=int, default=480)
    parser.add_argument("--action-dim", type=int, default=29)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256, 128])
    args = parser.parse_args()

    ckpt_path = os.path.abspath(args.checkpoint)
    if args.output is None:
        export_dir = os.path.join(os.path.dirname(ckpt_path), "exported")
        output_path = os.path.join(export_dir, "policy.onnx")
    else:
        export_dir = os.path.dirname(args.output)
        output_path = args.output

    os.makedirs(export_dir, exist_ok=True)

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    actor_state = ckpt["actor_state_dict"]

    # Infer dims from state dict if not overridden
    obs_dim = actor_state["mlp.0.weight"].shape[1]
    action_dim = actor_state["mlp.6.weight"].shape[0]
    print(f"[INFO] Detected obs_dim={obs_dim}, action_dim={action_dim}, hidden={args.hidden_dims}")

    model = ActorPolicy(obs_dim, action_dim, args.hidden_dims)
    model.load_state_dict(actor_state)
    model.eval()

    dummy_input = torch.zeros(1, obs_dim)
    with torch.no_grad():
        out = model(dummy_input)
    print(f"[INFO] Test forward pass output shape: {out.shape}")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,
        input_names=["obs"],
        output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
    )
    print(f"[INFO] ONNX policy saved to: {output_path}")

    # Quick verification
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(output_path)
        result = sess.run(["actions"], {"obs": np.zeros((1, obs_dim), dtype=np.float32)})
        print(f"[INFO] onnxruntime verification OK — output shape: {result[0].shape}")
    except ImportError:
        print("[INFO] onnxruntime not installed, skipping verification (pip install onnxruntime)")


if __name__ == "__main__":
    main()
