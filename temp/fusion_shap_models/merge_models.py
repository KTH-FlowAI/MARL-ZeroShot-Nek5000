"""

Merge two TD3 models by weight averaging.
    w_merged = alpha * w_cf + (1 - alpha) * w_pw
Both models must have the same architecture (same obs/action spaces).
Works best when both were fine-tuned from a common base model.
Usage:
    python merge_models.py
    python merge_models.py --alpha 0.8
    python merge_models.py --model_cf best_model_cf --model_pw best_model_pw --alpha 0.7
"""

import functools
import argparse, os
import numpy as np
from stable_baselines3 import TD3
import torch
from gym import spaces

act_space = spaces.Box(  low = -1.0,
                            high = 1.0,
                            shape = (1,),
                            dtype = np.float32)
    #### Define the observation space 
obs_space = spaces.Box(  low = -np.inf,
                            high = np.inf,
                            shape = (2,1,1),
                                dtype = np.float32)

def merge_td3_models(model_cf_path, model_pw_path, alpha, output_path):

    """Merge two TD3 models by interpolating their parameters."""

    print(f"Loading CF model: {model_cf_path}")
    model_cf = TD3.load(model_cf_path,custom_objects={"action_space": act_space, "observation_space": obs_space})
    print(f"Loading PW model: {model_pw_path}")

    model_pw = TD3.load(model_pw_path,custom_objects={"action_space": act_space, "observation_space": obs_space})
    # Get parameters from both models
    params_cf = model_cf.policy.state_dict()
    params_pw = model_pw.policy.state_dict()
    # Verify architectures match
    assert set(params_cf.keys()) == set(params_pw.keys()), "Model architectures don't match!"
    for key in params_cf:
        assert params_cf[key].shape == params_pw[key].shape,f"Shape mismatch for {key}: {params_cf[key].shape} vs {params_pw[key].shape}"
    # Merge: w = alpha * cf + (1 - alpha) * pw

    merged_params = {}

    for key in params_cf:
        merged_params[key] = alpha * params_cf[key] + (1 - alpha) * params_pw[key]
    # Load merged parameters into a new model (use cf as base)
    model_merged = TD3.load(model_cf_path,custom_objects={"action_space": act_space, "observation_space": obs_space})
    model_merged.policy.load_state_dict(merged_params)
    # Save as .zip explicitly
    if not output_path.endswith('.zip'):
        output_path = output_path + '.zip'
    model_merged.save(output_path)

    print(f"\nMerged model saved: {output_path}")
    print(f"  alpha = {alpha} (cf={alpha:.2f}, pw={1-alpha:.2f})")
    print(f"\nParameter comparison:")
    print(f"  {'Layer':<40s} {'CF norm':>10s} {'PW norm':>10s} {'Merged norm':>10s} {'Diff':>10s}")
    print(f"  {'-'*80}")
    for key in sorted(params_cf.keys()):

        cf_norm = params_cf[key].float().norm().item()

        pw_norm = params_pw[key].float().norm().item()

        merged_norm = merged_params[key].float().norm().item()

        diff = (params_cf[key].float() - params_pw[key].float()).norm().item()

        print(f"  {key:<40s} {cf_norm:>10.4f} {pw_norm:>10.4f} {merged_norm:>10.4f} {diff:>10.4f}")
    return model_merged

def main():

    parser = argparse.ArgumentParser(description='Merge two TD3 models by weight averaging')

    parser.add_argument('--model_cf', type=str, default='best_model_transf_cf',
                        help='Path to CF model (without .zip)')

    parser.add_argument('--model_pw', type=str, default='best_model_transf_pw',
                        help='Path to PW model (without .zip)')

    parser.add_argument('--alpha', type=float, default=0.8,
                        help='Blending weight: alpha*CF + (1-alpha)*PW (default: 0.8)')

    parser.add_argument('--output', type=str, default=None,
                        help='Output path (default: merged_alpha{alpha})')

    args = parser.parse_args()

    out_folder="./merged_models"
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
        print(f"Initialize the folder {out_folder}")

    if args.output is None:
        args.output = f'best_model_merged_alpha{args.alpha:.2f}.zip'
    output_model = os.path.join(out_folder,args.output)
    merge_td3_models(args.model_cf, args.model_pw, args.alpha, output_model)

if __name__ == '__main__':
    main()
