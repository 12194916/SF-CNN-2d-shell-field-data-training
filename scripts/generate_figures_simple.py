"""
Generate visualization figures for the trained multi_model_6
Simplified version - no OOD, uses our trained model
"""

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
import sys
sys.path.append('../utils')

import numpy as np
import matplotlib.pyplot as plt
import torch
from load_data import *
from evaluate import *
from visualize import *
from cnn_model import *

DIR = "../figures/"
EXT = ".png"
DPI = 300

def to_path(name):
    return DIR + name + EXT

def plot_stress_predictions():
    """Plot stress field predictions on test samples"""

    output_file = to_path("stress_predictions")
    print(f"Creating stress prediction figure: {output_file}")

    # Load model
    model = torch.load("../models/multi_model_6.pth")
    model.eval()

    # Load datasets
    print("Loading datasets...")
    datasets_vor = load_tr_te_od_data("../data/stress_vor_w.mat", "../data/stress_vor_o.mat", scale=10000)
    datasets_lat = load_tr_te_od_data("../data/stress_lat_w.mat", "../data/stress_lat_o.mat", scale=10000)

    datasets = dict()
    datasets['tr'] = datasets_vor['tr'] + datasets_lat['tr']
    datasets['te'] = datasets_vor['te'] + datasets_lat['te']

    # Evaluate on test set
    print("Evaluating on test set...")
    vals = eval_model_all(model, datasets["te"])
    order = np.argsort(vals)
    N = len(vals)

    # Select samples: best, median, worst
    ranks = [-1, N//2, 0]  # Best, median, worst
    selected_samples = [datasets["te"][order[rank]] for rank in ranks]

    print(f"Selected samples: Best R²={vals[order[-1]]:.3f}, Median R²={vals[order[N//2]]:.3f}, Worst R²={vals[order[0]]:.3f}")

    # Plot comparison
    plot_comparison(model, selected_samples, filename=output_file, dpi=DPI)
    print(f"Saved: {output_file}")

def plot_r2_distribution():
    """Plot R² distribution as box plots"""

    output_file = to_path("r2_boxplot")
    print(f"Creating R² distribution figure: {output_file}")

    # Load model
    model = torch.load("../models/multi_model_6.pth")
    model.eval()

    # Load datasets
    print("Loading datasets...")
    datasets_vor = load_tr_te_od_data("../data/stress_vor_w.mat", "../data/stress_vor_o.mat", scale=10000)
    datasets_lat = load_tr_te_od_data("../data/stress_lat_w.mat", "../data/stress_lat_o.mat", scale=10000)

    datasets = dict()
    datasets['tr'] = datasets_vor['tr'] + datasets_lat['tr']
    datasets['te'] = datasets_vor['te'] + datasets_lat['te']

    # Evaluate
    print("Evaluating model...")
    vals = eval_model_multiple(model, datasets)

    # Plot boxes
    plot_boxes(vals, filename=output_file, dpi=DPI)
    print(f"Saved: {output_file}")

def plot_r2_violin():
    """Plot R² distribution as violin plots"""

    output_file = to_path("r2_violin")
    print(f"Creating R² violin figure: {output_file}")

    # Load model
    model = torch.load("../models/multi_model_6.pth")
    model.eval()

    # Load datasets
    print("Loading datasets...")
    datasets_vor = load_tr_te_od_data("../data/stress_vor_w.mat", "../data/stress_vor_o.mat", scale=10000)
    datasets_lat = load_tr_te_od_data("../data/stress_lat_w.mat", "../data/stress_lat_o.mat", scale=10000)

    datasets = dict()
    datasets['tr'] = datasets_vor['tr'] + datasets_lat['tr']
    datasets['te'] = datasets_vor['te'] + datasets_lat['te']

    # Evaluate
    print("Evaluating model...")
    vals = eval_model_multiple(model, datasets)

    # Plot violins
    plot_violins(vals, filename=output_file, dpi=DPI)
    print(f"Saved: {output_file}")

if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING FIGURES FOR multi_model_6")
    print("=" * 60)
    print()

    os.makedirs(DIR, exist_ok=True)

    # Generate figures
    plot_stress_predictions()
    print()
    plot_r2_distribution()
    print()
    plot_r2_violin()

    print()
    print("=" * 60)
    print("ALL FIGURES GENERATED!")
    print(f"Output directory: {DIR}")
    print("=" * 60)
