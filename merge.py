from PIL import Image
from tqdm import tqdm
import numpy as np
import os

img_path = 'experiments/sampling'
np_path = 'experiments/sampling_noise'

merge = []
for i in tqdm(range(5000)):
    img = Image.open(os.path.join(img_path, f'{i}.png'))
    img = np.array(img) / 255.0
    img = 2 * img - 1

    npy = np.load(os.path.join(np_path, f'{i}.npy'))
    
    merge.append(np.concatenate((img[None, :], npy[None, :]), axis=0))

np.save('custom_data.npy', np.array(merge))