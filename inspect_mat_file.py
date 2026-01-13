"""
Script to inspect and visualize .mat file structure for SFP-CNN
This helps understand the format needed to create your own training data
"""

import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# Load a .mat file
mat_file = 'data/stress_vor_o.mat'
print(f"Loading: {mat_file}\n")
mat = io.loadmat(mat_file)

# Display all keys in the file
print("=" * 60)
print("KEYS IN .MAT FILE:")
print("=" * 60)
for key in mat.keys():
    if not key.startswith('__'):
        print(f"  {key}: {mat[key].shape}")
print()

# Get basic information
num_samples = len(mat['nodes'])
print("=" * 60)
print(f"DATASET INFORMATION:")
print("=" * 60)
print(f"Number of samples (meshes): {num_samples}")
print()

# Inspect first sample in detail
sample_idx = 0
print("=" * 60)
print(f"SAMPLE {sample_idx} DETAILS:")
print("=" * 60)

nodes = mat['nodes'][sample_idx, 0].T  # Shape: (num_nodes, 2)
elem = mat['elem'][sample_idx, 0].T - 1  # Shape: (num_elements, 3), -1 for 0-indexing
stress = mat['stress'][sample_idx, 0].flatten()  # Shape: (num_nodes,)
dt = mat['dt'][sample_idx, 0].flatten()  # Shape: (num_nodes,)
sdf = mat['sdf'][sample_idx, 0].T  # Shape: (64, 64)

print(f"Nodes shape: {nodes.shape}")
print(f"  - Number of nodes: {nodes.shape[0]}")
print(f"  - Coordinates: (x, y)")
print(f"  - X range: [{nodes[:, 0].min():.3f}, {nodes[:, 0].max():.3f}]")
print(f"  - Y range: [{nodes[:, 1].min():.3f}, {nodes[:, 1].max():.3f}]")
print()

print(f"Elements shape: {elem.shape}")
print(f"  - Number of triangular elements: {elem.shape[0]}")
print(f"  - Each element: 3 node indices")
print(f"  - Element connectivity example: {elem[0]}")
print()

print(f"Stress values shape: {stress.shape}")
print(f"  - One value per node: {stress.shape[0]} values")
print(f"  - Stress range: [{stress.min():.6f}, {stress.max():.6f}]")
print(f"  - This is your TARGET to predict!")
print()

print(f"Distance Transform (dt) shape: {dt.shape}")
print(f"  - One value per node: {dt.shape[0]} values")
print(f"  - DT range: [{dt.min():.6f}, {dt.max():.6f}]")
print(f"  - Distance to boundary for each node")
print()

print(f"Signed Distance Field (SDF) shape: {sdf.shape}")
print(f"  - Regular grid: {sdf.shape[0]} x {sdf.shape[1]}")
print(f"  - SDF range: [{sdf.min():.3f}, {sdf.max():.3f}]")
print(f"  - This is the CNN INPUT (shape representation)")
print()

# Visualize the sample
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle(f'Sample {sample_idx} from {mat_file}', fontsize=16, fontweight='bold')

# 1. Mesh structure (nodes + elements)
ax = axes[0, 0]
triang = Triangulation(nodes[:, 0], nodes[:, 1], elem)
ax.triplot(triang, 'k-', linewidth=0.5, alpha=0.3)
ax.plot(nodes[:, 0], nodes[:, 1], 'ro', markersize=2)
ax.set_aspect('equal')
ax.set_title(f'Mesh Structure\n{nodes.shape[0]} nodes, {elem.shape[0]} triangles')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.grid(True, alpha=0.3)

# 2. Stress field on mesh (TARGET)
ax = axes[0, 1]
tcf = ax.tricontourf(triang, stress, levels=20, cmap='jet')
ax.triplot(triang, 'k-', linewidth=0.3, alpha=0.2)
plt.colorbar(tcf, ax=ax, label='Stress (Target)')
ax.set_aspect('equal')
ax.set_title('Stress Field (Ground Truth)\nThis is what the model predicts!')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# 3. Distance Transform
ax = axes[0, 2]
tcf = ax.tricontourf(triang, dt, levels=20, cmap='viridis')
ax.triplot(triang, 'k-', linewidth=0.3, alpha=0.2)
plt.colorbar(tcf, ax=ax, label='Distance Transform')
ax.set_aspect('equal')
ax.set_title('Distance Transform\nDistance to boundary')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# 4. Signed Distance Field (CNN INPUT)
ax = axes[1, 0]
im = ax.imshow(sdf, cmap='RdBu_r', origin='lower', extent=[0, 1, 0, 1])
plt.colorbar(im, ax=ax, label='SDF')
ax.set_title('Signed Distance Field (SDF)\nCNN Input - Regular 64x64 grid')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# 5. Node positions on top of SDF
ax = axes[1, 1]
im = ax.imshow(sdf, cmap='RdBu_r', origin='lower', extent=[0, 1, 0, 1], alpha=0.7)
ax.scatter(nodes[:, 0], nodes[:, 1], c=stress, s=10, cmap='jet', edgecolors='black', linewidth=0.3)
plt.colorbar(im, ax=ax, label='SDF')
ax.set_title('Mesh Nodes overlaid on SDF\nShows relationship between grid and mesh')
ax.set_xlabel('X')
ax.set_ylabel('Y')

# 6. Summary text
ax = axes[1, 2]
ax.axis('off')
summary_text = f"""
DATA FORMAT SUMMARY:

Input to CNN:
  • SDF: {sdf.shape} grid
  • Represents shape geometry

Mesh Information:
  • Nodes: {nodes.shape[0]} points (x, y)
  • Elements: {elem.shape[0]} triangles
  • Each element = 3 node indices

Target Output:
  • Stress: {stress.shape[0]} values
  • One per node
  • Range: [{stress.min():.4f}, {stress.max():.4f}]

Additional Features:
  • Distance Transform (dt)
  • One value per node

HOW IT WORKS:
1. CNN processes 64x64 SDF grid
2. Features interpolated to mesh nodes
3. MLP predicts stress at each node
"""
ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
        verticalalignment='center')

plt.tight_layout()
plt.savefig('mat_file_visualization.png', dpi=150, bbox_inches='tight')
print("=" * 60)
print("Visualization saved as: mat_file_visualization.png")
print("=" * 60)
plt.show()

# Print detailed format requirements
print("\n" + "=" * 60)
print("CREATING YOUR OWN .MAT FILE:")
print("=" * 60)
print("""
To create a compatible .mat file for training, you need:

1. NODES: (N, 1) array of (2, num_nodes) arrays
   - Each entry: 2D coordinates [x, y] for all nodes
   - Example: nodes[i,0] = [[x1,x2,...], [y1,y2,...]]

2. ELEM: (N, 1) array of (3, num_elements) arrays
   - Each entry: triangle connectivity (1-indexed!)
   - Example: elem[i,0] = [[n1,n2,n3], [n4,n5,n6], ...]

3. STRESS/TEMP: (N, 1) array of (num_nodes, 1) arrays
   - Your FEM results / scalar field values
   - One value per node

4. DT: (N, 1) array of (num_nodes, 1) arrays
   - Distance transform values
   - Distance from each node to boundary

5. SDF: (N, 1) array of (64, 64) arrays
   - Signed distance field on regular grid
   - Positive inside, negative outside

Example code to create .mat file:
----------------------------------------
import scipy.io as io
import numpy as np

data = {
    'nodes': np.array([[your_nodes_array]], dtype=object),
    'elem': np.array([[your_elements_array]], dtype=object),
    'stress': np.array([[your_stress_values]], dtype=object),
    'dt': np.array([[your_dt_values]], dtype=object),
    'sdf': np.array([[your_sdf_grid]], dtype=object)
}
io.savemat('my_data.mat', data)
----------------------------------------

Key Requirements:
✓ 2D meshes only (x, y coordinates)
✓ Triangular elements
✓ Consistent node indexing
✓ SDF must be 64x64 grid
✓ All coordinates normalized to [0, 1]
""")

print("\n" + "=" * 60)
print("To inspect other samples, modify sample_idx at line 33")
print("To inspect different files, modify mat_file at line 12")
print("=" * 60)
