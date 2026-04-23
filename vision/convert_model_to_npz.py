#!/usr/bin/env python3
"""
Convert a trained .pt model to .weights.npz for Jetson inference (no torch needed).

Usage:
    python convert_model_to_npz.py                    # defaults
    python convert_model_to_npz.py --input models/nn_3d.pt --output models/nn_3d.weights.npz
"""
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convert .pt to .weights.npz")
    parser.add_argument("--input", default="models/nn_3d.pt", help="Input .pt file")
    parser.add_argument("--output", default=None, help="Output .npz file (default: input.weights.npz)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.weights.npz')

    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    import torch
    import numpy as np

    state_dict = torch.load(input_path, map_location='cpu')

    # DeltaAccelNN3D architecture: net.0=Linear(12,32), net.2=Linear(32,3)
    W1 = state_dict['net.0.weight'].numpy()  # (32, 12)
    b1 = state_dict['net.0.bias'].numpy()    # (32,)
    W2 = state_dict['net.2.weight'].numpy()  # (3, 32)
    b2 = state_dict['net.2.bias'].numpy()    # (3,)

    np.savez(output_path, W1=W1, b1=b1, W2=W2, b2=b2)
    print(f"Converted {input_path} -> {output_path}")
    print(f"  W1: {W1.shape}, b1: {b1.shape}, W2: {W2.shape}, b2: {b2.shape}")
    print(f"  Total params: {W1.size + b1.size + W2.size + b2.size}")


if __name__ == "__main__":
    main()
