import os
import scipy.io as sio
import shutil

DEVKIT_DIR = 'ILSVRC2012/ILSVRC2012_devkit_t12'
VAL_DIR    = 'ILSVRC2012/ILSVRC2012_img_val'


# 1. Load the .mat
meta = sio.loadmat(os.path.join(DEVKIT_DIR, 'data', 'meta.mat'), squeeze_me=True)
synsets = meta['synsets']  # this is a NumPy structured array

# 2. Build id→WNID map
#    synsets['ILSVRC2012_ID'] is an array of ints, synsets['WNID'] is an array of bytes (or str)
ids   = synsets['ILSVRC2012_ID'].astype(int)
wnids = synsets['WNID'].astype(str)
id2wnid = {i: w for i, w in zip(ids, wnids)}

# 3. Read the ground‐truth labels
gt_file = os.path.join(DEVKIT_DIR, 'data',
                       'ILSVRC2012_validation_ground_truth.txt')
with open(gt_file) as f:
    gt_labels = [int(x.strip()) for x in f]

# 4. Sort and move
val_files = sorted(f for f in os.listdir(VAL_DIR) if f.endswith('.JPEG'))
for fname, label in zip(val_files, gt_labels):
    wnid = id2wnid[label]
    dst  = os.path.join(VAL_DIR, wnid)
    os.makedirs(dst, exist_ok=True)
    shutil.move(os.path.join(VAL_DIR, fname), os.path.join(dst, fname))

print("Done: validation images reorganized into class subfolders under", VAL_DIR)