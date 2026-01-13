"""
Train only the main multi_model_6 (combined stress model)
This is the best model for testing before using custom data
"""

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
import sys
sys.path.append('../utils')

import torch
import numpy as np
from load_data import *
from training import *
from evaluate import *
from cnn_model import *

def train_single_stress_model():
    """Train the main combined stress model (best performance)"""

    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 60)
    print("TRAINING: multi_model_6.pth (Combined Stress Model)")
    print("=" * 60)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("Dataset: Voronoi + Lattice shapes")
    print("Training samples: ~1600")
    print("Architecture: 6 pooling layers, 20 filters")
    print("Expected time: 5-10 minutes on RTX 3060 Ti")
    print("=" * 60)
    print()

    # Load both Voronoi and Lattice datasets (no OOD)
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

    # Create model and move to GPU
    print("Creating model...")
    model = MultiNet(kernel_size=5, num_layers=6, num_filters=20)
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print()

    # Move data to GPU (done automatically in training loop)
    print("Starting training (50 epochs)...")
    print("Note: Data will be moved to GPU batch-by-batch during training")
    print()
    result = train_model(model, dataset=datasets["tr"], valset=datasets["te"], epochs=50)

    # Save model
    os.makedirs("../models", exist_ok=True)
    torch.save(model, "../models/multi_model_6.pth")
    print()
    print("=" * 60)
    print("Model saved: models/multi_model_6.pth")
    print(f"Training time: {result['time']:.2f} minutes")
    print("=" * 60)
    print()

    # Evaluate on training and testing datasets
    print("Evaluating model performance...")
    vals = eval_model_multiple(model, datasets)

    print()
    print("=" * 60)
    print("RESULTS:")
    print("=" * 60)
    print(f"Training R²   (median): {np.median(vals['tr']):.3f}")
    print(f"Testing R²    (median): {np.median(vals['te']):.3f}")
    print("=" * 60)
    print()
    print("SUCCESS! Model is ready to use.")
    print("Next step: Create your custom .mat file and test predictions")
    print("=" * 60)

if __name__ == "__main__":
    train_single_stress_model()
