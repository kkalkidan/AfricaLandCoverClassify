import math
import os
from osgeo import gdal
from datetime import datetime
from pathlib import Path

#Save subset ROI to given path
def subsetGeoTiff(ds, outFileName, arr_out, start, size, bands, outDir ):

    driver = gdal.GetDriverByName("GTiff")
    #set compression
    
    if('label' in outDir):
        outdata = driver.Create(outFileName, size[0], size[1], bands, gdal.GDT_Byte)
    else:
        outdata = driver.Create(outFileName, size[0], size[1], bands, gdal.GDT_Float64)
    newGeoTransform = list( ds.GetGeoTransform() )
    newGeoTransform[0] = newGeoTransform[0] + start[0]*newGeoTransform[1] + start[1]*newGeoTransform[2]
    newGeoTransform[3] = newGeoTransform[3] + start[0]*newGeoTransform[4] + start[1]*newGeoTransform[5]

    outdata.SetGeoTransform( newGeoTransform )    
    outdata.SetProjection(ds.GetProjection())

    for i in range(0,bands) :
    
        outdata.GetRasterBand(i+1).WriteArray(arr_out[i, :, :])
        outdata.GetRasterBand(i+1).SetNoDataValue(0)

    outdata.FlushCache()
# batch_size = ""

def crop(image, outDir, filename, file_extension, ds):
    imageWidth = image.shape[-1]
    imageHeight = image.shape[-2]
    tileSizeX = 32
    tileSizeY = 32

    offsetX = int(tileSizeX)
    offsetY = int(tileSizeY)

    tileSize = (tileSizeY, tileSizeX)
    
    for startX in range(0, imageWidth, offsetX):
        for startY in range(0, imageHeight, offsetY):
            endX = startX + tileSizeX
            endY = startY + tileSizeY
            currentTile = image[:, startX:endX,startY:endY]
            #if you want to save save directly with opencv
            # However reverse order of data
            #cv2.imwrite(filename + '_%d_%d' % (nTileY, nTileX)  + file_extension, currentTile)
            start = (startY,startX)
            outFullFileName = os.path.join( outDir, filename + '_%d_%d' % (startY, startX)  + file_extension)
            # print(outFullFileName)
            # print(currentTile.shape)
            subsetGeoTiff(ds, outFullFileName, currentTile, start, tileSize,  currentTile.shape[0], outDir)            
    
    

def lcn_crop(data_loader=None, file_path=None):
    startTime = datetime.now()
    print("Starting" , datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
    for x, y, tile_name, tile_date, path, season in data_loader:
        # print(path)
        # print(x.shape, y.shape)
        ds = gdal.Open(path[0])
        
        # print(x.shape, y.shape, tile_name, season)
        for image, outDir in zip([x[0], y[0]], ['images', 'labels']): 
            filename = tile_name[0] + 's' + season[0]
            #TODO add if 'labels' in outDir:
            #               filename = tile_name[0].replace('median', 'landcover')
            if outDir == 'labels':
                filename = tile_name[0]
            
            outDir = f'afrimap/data_preparation/lcn32_dataset_{outDir}'
            file_extension = '.tif'
            # print(image.shape)
            if not os.path.exists(outDir):
                os.makedirs(outDir)
            crop(image.numpy(), outDir, filename, file_extension, ds)    
    endTime = datetime.now()
    ds=None
    print("Finished " , datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), " in  " , endTime-startTime)     
    print("Cropped data written to: \
    afrimap/data_preparation/lcn32_dataset_images and \
    afrimap/data_preparation/lcn32_dataset_labels", )

def infer_crop(file_path):

    startTime = datetime.now()
    print("Starting" , datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))

    ds = gdal.Open(str(file_path))
    image = ds.ReadAsArray() #[:, 256:4000, 256:4000]

    shape = image.shape

    if(len(shape) == 2):
        image = image.reshape((1, shape[0], shape[1]))
    print(image.shape)
    outDir = Path(file_path).stem

    file_extension = '.tif'

    if not os.path.exists(outDir):
        os.makedirs(outDir)

    filename = outDir

    crop(image, outDir, filename, file_extension, ds)  
    endTime = datetime.now()
    ds=None
    print("Finished " , datetime.now().strftime("%m/%d/%Y, %H:%M:%S"), " in  " , endTime-startTime)

    os.system(f"rm -r --force afrimap/data_preparation/{outDir}; mv {outDir} afrimap/data_preparation/")

    dest = str(Path("afrimap/data_preparation", outDir))
    print("Cropped data written to: ", dest)

    return dest