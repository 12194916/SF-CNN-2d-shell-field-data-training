import scipy.io as io
import os

print('Dataset Sizes:')
print('='*60)

files = [
    'stress_vor_o.mat',
    'stress_vor_w.mat',
    'stress_lat_o.mat',
    'stress_lat_w.mat',
    'temp_vor_o.mat',
    'temp_vor_w.mat',
    'temp_lat_o.mat',
    'temp_lat_w.mat'
]

total = 0
for f in files:
    path = f'data/{f}'
    if os.path.exists(path):
        mat = io.loadmat(path)
        n = len(mat['nodes'])
        total += n
        print(f'{f:25s}: {n:4d} samples')
    else:
        print(f'{f:25s}: NOT FOUND')

print('='*60)
print(f'TOTAL SAMPLES: {total}')
print()

# Show what "within-distribution" and "out-of-distribution" means
print('Dataset Types:')
print('='*60)
print('*_w.mat files: Within-distribution (WSS)')
print('  - Used for TRAINING and TESTING')
print('  - Split 80% train, 20% test')
print()
print('*_o.mat files: Out-of-distribution (OSS)')
print('  - Used for OOD TESTING only')
print('  - Tests generalization to different shapes')
print()
print('vor = Voronoi-based shapes')
print('lat = Lattice-based shapes')
print('='*60)
