"""
Evaluate an already trained model
Use this after training completes to check performance
"""

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
import sys
sys.path.append('../utils')

import torch
import numpy as np
from load_data import *
from evaluate import *

def evaluate_model():
    """Evaluate the trained multi_model_6.pth"""

    print("=" * 60)
    print("EVALUATING TRAINED MODEL")
    print("=" * 60)

    # Load trained model
    model_path = "../models/multi_model_6.pth"

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Train the model first using train_single_model.py")
        return

    print(f"Loading model: {model_path}")
    model = torch.load(model_path)
    model.eval()

    # Check device
    device = next(model.parameters()).device
    print(f"Model device: {device}")
    print()

    # Load datasets (only training and testing, no OOD)
    print("Loading datasets...")
    datasets_vor = load_tr_te_od_data("../data/stress_vor_w.mat", "../data/stress_vor_o.mat", scale=10000)
    datasets_lat = load_tr_te_od_data("../data/stress_lat_w.mat", "../data/stress_lat_o.mat", scale=10000)

    # Combine them (only tr and te, skip od)
    datasets = dict()
    datasets['tr'] = datasets_vor['tr'] + datasets_lat['tr']
    datasets['te'] = datasets_vor['te'] + datasets_lat['te']

    print(f"Training samples: {len(datasets['tr'])}")
    print(f"Testing samples: {len(datasets['te'])}")
    print()

    # Evaluate on training and testing only
    print("Evaluating model performance...")
    print("(This may take a few minutes...)")
    print()

    vals = eval_model_multiple(model, datasets)

    print()
    print("=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Training R²   (median): {np.median(vals['tr']):.3f}")
    print(f"Training R²   (mean):   {np.mean(vals['tr']):.3f}")
    print(f"Training R²   (min):    {np.min(vals['tr']):.3f}")
    print(f"Training R²   (max):    {np.max(vals['tr']):.3f}")
    print()
    print(f"Testing R²    (median): {np.median(vals['te']):.3f}")
    print(f"Testing R²    (mean):   {np.mean(vals['te']):.3f}")
    print(f"Testing R²    (min):    {np.min(vals['te']):.3f}")
    print(f"Testing R²    (max):    {np.max(vals['te']):.3f}")
    print("=" * 60)
    print()
    print("SUCCESS! Model evaluation complete.")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_model()
