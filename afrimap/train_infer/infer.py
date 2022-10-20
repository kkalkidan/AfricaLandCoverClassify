                

from pathlib import Path
from afrimap.train_infer.utils import create_configs, get_train_val, prepare_device
import torch 
from osgeo import gdal
from tqdm import tqdm
import os

def infer(model_path, dataset_path):

    device, device_id = prepare_device(1)
    model, _, _, _, _ = create_configs()
    model = model.to(device)
  
    model.load_state_dict(torch.load(model_path))

    dataset_loader = get_train_val(dataset_path)
   
    model.eval()
    bands = 1

    with torch.no_grad():
         # run each batch in the train_loader 
        for data, _, _, _, label_path in tqdm(dataset_loader):
            
            # print('label_path', label_path)
            data = data.to(device)
            # predict input 
            output = model(data)
            label_path = label_path[0]
            B, C, H, W = output.shape
            # print('shape', output.shape)
            output = output.argmax(dim=1)
            # print('after argmax', output.shape)
            filename = Path(label_path).stem
            ds = gdal.Open(label_path)

            file_extension = '.tif'

            driver = gdal.GetDriverByName("GTiff")
            outDir = Path(label_path).stem.split('_')
            outDir = outDir[0] + outDir[1] 
            if not os.path.exists(outDir):
                os.makedirs(outDir)
            outFullFileName = str(Path(outDir, (filename + file_extension)))
            outdata = driver.Create(outFullFileName, H, W, bands, gdal.GDT_Byte)
            newGeoTransform = list( ds.GetGeoTransform() )
 

            outdata.SetGeoTransform( newGeoTransform )    
            outdata.SetProjection(ds.GetProjection())

            for i in range(0,bands) :
            
                outdata.GetRasterBand(i+1).WriteArray(output[i, :, :].cpu().numpy())
                outdata.GetRasterBand(i+1).SetNoDataValue(0)
            outdata.FlushCache()
    os.system(f"mv {outDir} afrimap/post_process/")
          


   

