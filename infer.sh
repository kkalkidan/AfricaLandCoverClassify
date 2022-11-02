#!/bin/bash


# downloads the satellite imagery from Google Earth Engine
# I am commenting out this section because it required access to google
# python3.8 -m afrimap collect-data --type infer --xmin 35.906660425697865  --ymin 0.29520872039559976 --xmax 35.95278042569787  --ymax 0.34132872039559975 --start_date 2020-01-01 --end_date  2020-12-30
# the download is saved in afrimap/data_collection/infer_data

SAT_IMAGE=$1

echo "Cropping the satellite image into 32x32 pixel image"
echo "###### Cropping: $SAT_IMAGE.tif #######"
python -m  afrimap prep-data --type infer --image_path afrimap/data_collection/infer_data/${SAT_IMAGE%.*}.tif

echo "####### Running Inference ... #######"
python -m afrimap train-infer --type infer --image_path afrimap/data_preparation/${SAT_IMAGE%.*} --model_path 22_0200_8class_manet_kcl_best.pth

echo "###### Mosaic and save tif and png file at ${SAT_IMAGE%.*} #######"
python -m afrimap post-process --predictions afrimap/post_process/infer_output --destination "${SAT_IMAGE%.*}_seg.tif"