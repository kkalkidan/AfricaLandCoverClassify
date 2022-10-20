# African LandCover Classification Using MA-Net-Like Model

## Introduction

Landcover map is essential for detecting land cover changes caused by natural and human factors. This package, given the geometry of an area of interest within Africa and satellite image acquistion start and end date for Sentinel-2 Level-2A satellite image data, returns landcover classification map for the area of interest. 

Input:
- Geometry of area of interest Xmin, Ymin, Xmax, Ymax
- Acquistion start and end date(the package export the median of the image collection acquired within the specified start and end date)

Returns:
- Landcover map with seven land cover classes 
![lcn_classes.png](https://github.com/kkalkidan/AfricaLandCoverClassify/blob/main/afrimap/lcn_classes.png)

## Modules

The Afrimap package is consists of four modules:
1. Data Collection Module
2. Data Preparation Module
3. Training and Inference Module
4. Post-processing Module
![afrimap_modules.png](https://github.com/kkalkidan/AfricaLandCoverClassify/blob/main/afrimap/afrimap_modules.png)
The first three modules have two phases or types, training and inference phase. And, the last module is run only during the inference phase.
```
Usage: afrimap [OPTIONS] COMMAND [ARGS]...

  Command to download, preprocess, train and run inference for landcover
  classification of Africa.

Options:
  --help  Show this message and exit.

Commands:
  collect-data
  post-process
  prep-data
  train-infer
```
### 1. Data Collection
```
Usage: afrimap collect-data [OPTIONS]

Options:
  --type [train|infer]  choose task type, train for training, infer for
                        inference  [required]
  --xmin FLOAT          xMin of xMin, yMin, xMax, yMax for area of interest
  --xmax FLOAT          xMax of xMin, yMin, xMax, yMax for area of interest
  --ymin FLOAT          yMin of xMin, yMin, xMax, yMax for area of interest
  --ymax FLOAT          xMax of xMin, yMin, xMax, yMax for area of interest
  --start_date TEXT     start date for filtering collection by date in YYYY-
                        mm-dd format
  --end_date TEXT       end date for filtering collection by date in YYYY-mm-
                        dd format
  --mlhub_key TEXT      MLHUB_API_KEY from Radiant
                        [MLHUB](https://mlhub.earth/profile)
  --help                Show this message and exit.
```
#### Training

During the training phase, the package download LandCoverNet Africa dataset from [Radient MLHub](https://mlhub.earth/data/ref_landcovernet_af_v1) 

This step requires a MLHUB_API_KEY from Radient Hub. After registering for an account an api key can be acquired from [MLHub Dashboard](https://dashboard.mlhub.earth/)

```
python -m afrimap collect-data --type train --mlhub_key MLHUB_KEY  

```

The train data will be downloaded to the `data_collection/train_data folder`. The dataset is more than 80GB. For running this package as a test we have provided a smaller mock dataset that has similar structure as the original LandCoverNet Africa dataset. 

#### Inference

In the inference phase, Sentinel-2 Level-2A images acquired within the specified start and end dates are download from Google Earth Engine data catalog. 

This step requires a Google Earth Engine account. Please signup by following this [link](https://earthengine.google.com/new_signup/).

                                                                                   
```
earthengine authenticate --auth_mode notebook -quiet
// authorize access to your Earth Engine account by coping and pasting the provided command with the auth code

python -m afrimap collect-data --type infer --xmin 11.489628  --ymin 8.62211116 --xmax 17.993535  --ymax 13.9637367 --start_date 2020-01-01 --end_date  2020-12-30

```
The downloaded image will be exported to Google Drive satellite `sentinel2_level2_images` folder.

### 2. Data Preparation Module
```
Usage: afrimap prep-data [OPTIONS]

Options:
  --type [train|infer]  choose task type, train for training, infer for
                        inference  [required]
  --image_path TEXT     path to the folder containing the satellite image for
                        the dataset  [required]
  --label_path TEXT     path to the folder containing the ground truth or
                        labels(needed only for training)
  --help                Show this message and exit.

```
#### Training 

After unziping the downloaded files during the data collection stage, provide the path to folder containing satellite images and training label dataset. 

we can use images and labels in  `afrimap/data_collection/mock_raw_data` as an example
```
python -m afrimap prep-data --type train  --image_path  afrimap/data_collection/mock_raw_data/images --label_path afrimap/data_collection/mock_raw_data/labels

```

The prep-data writes the cropped images and labels in `afrimap/data_preparation/lcn32_dataset_images/labels` folders. It also creates train and val csv file containing the list of cropped images with their respective labels. 

#### Inference 

For the inference phase, the use only needs to provide the path to the median satellite image downloaded from Google Earth Engine. The satellite image is a tiff file with 13 bands.

```
python -m  afrimap prep-data --type infer --image_path afrimap/data_collection/Botswana_Gaborone_median.tif 

```

The 32x32 cropped images will be written to `afrimap/data_preparation/<image file name>` folder. It also creates csv file containing the list of cropped images in the dataset folder.

### 3. Training and Inference Module

This part of the module handles training and inference using the preprocessed datasets passed from previous modules. 

```
Usage: afrimap train-infer [OPTIONS]

Options:
  --type [train|infer]  choose task type, train for training, infer for
                        inference  [required]
  --image_path TEXT     path to the folder containing the cropped satellite image for the dataset  [required]
  --label_path TEXT     path to the folder containing the groundtruth or
                        labels
  --nb_epochs INTEGER   number of epoches for training
  --lr FLOAT            learning rate
  --model_path TEXT     path to pre-trained model
  --help                Show this message and exit.
```

#### Training 

Train with the sample mock data(not recommended, only for testing purposes)

```
python -m afrimap train-infer --type train --image_path afrimap/data_preparation/lcn32_dataset_images --label_path afrimap/data_preparation/lcn32_dataset_labels --nb_epochs 5

```
The trained model will be saved in the `afrimap/train_infer/output` folder

#### Inference 
In the inference phase, the model path needs to be provided.

```
python -m afrimap train-infer --type infer --image_path afrimap/data_preparation/Botswana_Gaborone_median --model_path afrimap/train_infer/output/manet_best.pth

```
The segementation output will be saved in a folder inside `afrimap/post_process`

### 4. Post-processing Module

#### Inference 

This module mosaics the cropped segementation output produced by the training and inference module 
```
Usage: afrimap post-process [OPTIONS]

Options:
  --predictions TEXT  path to the folder containing the inferred tif files
                      [required]
  --destination TEXT  destination of the mosaic tiff file
  --help              Show this message and exit.
```

```
python -m afrimap post-process --predictions afrimap/post_process/BotswanaGaborone --destination BotswanaGaborone_seg
```


How to output a segementation map, given sample satellite image data?

> Make sure the satellite images are composed of 13 bands 
"B02", "B03","B04", "B05", "B06", "B07", "B08", "B8A","B11", "B12", "NDVI","MNDWI", "NDBI","SCL"

```

```
