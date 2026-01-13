# Training Information

## Quick Start - Train Single Model

```bash
cd d:\Research\sfp-cnn
conda activate multires_env
python scripts/train_single_model.py
```

## What This Script Does

### Model: `multi_model_6.pth`
- **Architecture**: Multi-resolution CNN with 6 pooling layers
- **Purpose**: Predicts stress fields on 2D meshes
- **Best overall performance**: R² ≈ 0.91

### Training Data

The script trains on **ALL available stress data**:

#### Datasets Used:
1. **stress_vor_w.mat** - 1000 Voronoi shapes (within-distribution)
2. **stress_lat_w.mat** - 1000 Lattice shapes (within-distribution)
3. **stress_vor_o.mat** - 200 Voronoi shapes (out-of-distribution test)
4. **stress_lat_o.mat** - 160 Lattice shapes (out-of-distribution test)

#### Split:
- **Training**: 1600 geometries (80% of vor_w + lat_w)
- **Validation**: 400 geometries (20% of vor_w + lat_w)
- **OOD Testing**: 360 geometries (all *_o.mat files)

### GPU Usage

✅ **Automatically uses GPU if available**
- Model moved to GPU: `model.to(device)`
- Data moved to GPU batch-by-batch during training
- Displays GPU info at start:
  - Device name (e.g., RTX 3060 Ti)
  - Available memory

### Training Time

- **GPU (RTX 3060 Ti)**: ~5-10 minutes
- **CPU**: ~60-70 minutes
- **Epochs**: 50
- **Optimizer**: Adam (lr=0.001)

### Output

Model saved to: `models/multi_model_6.pth`

### Expected Results

```
Training R²   (median): 0.925
Testing R²    (median): 0.911
OOD R²        (median): 0.881
```

## What is R²?

R² (R-squared) measures prediction accuracy:
- **1.0** = Perfect prediction
- **0.9** = Very good (90% variance explained)
- **0.8** = Good
- **< 0.5** = Poor

## Next Steps After Training

1. **Test the model**: Use `inspect_mat_file.py` to understand data format
2. **Create custom data**: Prepare your 2D shell FEM results in .mat format
3. **Run predictions**: Load the trained model and predict on your data

## Files Modified

✅ `scripts/train_single_model.py` - Training script with GPU support
✅ `utils/training.py` - Updated to move data to GPU automatically

## Training on Different Data

To train on temperature data instead:
```python
# In train_single_model.py, change line 40-41:
datasets_vor = load_tr_te_od_data("../data/temp_vor_w.mat", "../data/temp_vor_o.mat", scale=1)
datasets_lat = load_tr_te_od_data("../data/temp_lat_w.mat", "../data/temp_lat_o.mat", scale=1)
```
