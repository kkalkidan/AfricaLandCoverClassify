BANDS = ["B02", #0
        "B03", #1
        "B04", #2
        "B05", #3
        "B06", #4
        "B07", #5
        "B08", #6
        "B8A", #7
        "B11", #8 
        "B12", #9
        "NDVI", #10
        "MNDWI", #11
        "NDBI", #12
        "SCL"
        ]

MEAN = [1368.1, #2
        1671.2, #3
        1929.8, #4
        2324.7, #5
        2847.1, #6
        3086.2,
        3130.4,
        3269.8,
        3288.5,
        2583.9,
        0,
        0,
        0]

VARIANCE = [1899.3, #2
            1746.1, #3
            1753.1, #4
            1786.3, #5
            1547.1, #6
            1497.0,
            1473.4,
            1447.6,
            1533.8,
            1511.4,
            1,
            1,
            1]

LCN_IMAGES = 'afrimap/data_preparation/lcn32_dataset_images'
LCN_LABELS = 'afrimap/data_preparation/lcn32_dataset_labels'
CLIMATE_CHIP_NAME_MAP = 'afrimap/chip_name_climate_zone_mapping.csv'