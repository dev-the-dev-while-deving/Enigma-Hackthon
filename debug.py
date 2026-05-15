import numpy as np
import glob
for f in glob.glob('data_*.npz'):
    data = np.load(f)
    print(f, "Targets Max:", np.max(data['targets']), "Targets Min:", np.min(data['targets']))
