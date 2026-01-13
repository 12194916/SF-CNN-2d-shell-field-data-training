"""
Print evaluation tables for the trained multi_model_6
Simplified version - no OOD, uses our trained model
"""

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
import sys
sys.path.append('../utils')

import numpy as np
import torch
from load_data import *
from evaluate import *

def generate_r2_table():
    """Generate R² table for trained multi_model_6"""

    print("=" * 60)
    print("GENERATING R² TABLE")
    print("=" * 60)

    # Load model
    model_path = "../models/multi_model_6.pth"
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Train the model first using train_single_model.py")
        return

    print(f"Loading model: {model_path}")
    model = torch.load(model_path)
    model.eval()

    # Load datasets (no OOD)
    print("Loading datasets...")
    datasets_vor = load_tr_te_od_data("../data/stress_vor_w.mat", "../data/stress_vor_o.mat", scale=10000)
    datasets_lat = load_tr_te_od_data("../data/stress_lat_w.mat", "../data/stress_lat_o.mat", scale=10000)

    datasets = dict()
    datasets['tr'] = datasets_vor['tr'] + datasets_lat['tr']
    datasets['te'] = datasets_vor['te'] + datasets_lat['te']

    print(f"Training samples: {len(datasets['tr'])}")
    print(f"Testing samples: {len(datasets['te'])}")
    print()

    # Evaluate
    print("Evaluating model...")
    vals = eval_model_multiple(model, datasets)

    # Create table
    table_string = """
R² Performance Table - multi_model_6
=====================================

Dataset Split    | Median R²  | Mean R²   | Min R²   | Max R²
-----------------|------------|-----------|----------|----------
Training         | {:.3f}     | {:.3f}    | {:.3f}   | {:.3f}
Testing          | {:.3f}     | {:.3f}    | {:.3f}   | {:.3f}

Total Samples: {} training, {} testing
Model: ../models/multi_model_6.pth
"""

    output = table_string.format(
        np.median(vals['tr']), np.mean(vals['tr']), np.min(vals['tr']), np.max(vals['tr']),
        np.median(vals['te']), np.mean(vals['te']), np.min(vals['te']), np.max(vals['te']),
        len(datasets['tr']), len(datasets['te'])
    )

    # Print to console
    print()
    print(output)

    # Save to file
    os.makedirs("../figures", exist_ok=True)
    output_file = "../figures/r2_table_multi_model_6.txt"
    with open(output_file, "w") as f:
        f.write(output)

    print(f"Table saved to: {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    generate_r2_table()
