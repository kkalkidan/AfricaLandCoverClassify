from afrimap.data_preparation.cropper import infer_crop

import os
import pandas as pd

def infer_prep(image_path):
    outDir = infer_crop(image_path)
    layers = os.listdir(outDir)
    df = pd.DataFrame({"image_path": layers})
    df.to_csv("afrimap/train_infer/infer.csv", index=False)