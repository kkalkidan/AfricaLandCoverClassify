import ee
import datetime

# ee.Authenticate(auth_mode='notebook', code_verifier='')
ee.Initialize()

MAX_CLOUD_PROBABILITY = 50
s2 = ee.ImageCollection("COPERNICUS/S2_SR")
s2Clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY');

def maskEdges(s2_img):
      return s2_img.updateMask(
      s2_img.select('B8A').mask().updateMask(s2_img.select('B9').mask()));


def maskClouds(img):
    clouds = ee.Image(
    img.get('cloud_mask')).select('probability');
    isNotCloud = clouds.lt(MAX_CLOUD_PROBABILITY);
    return img.updateMask(isNotCloud);

def download(image, crs, file_name, region):
      folder = 'satellite_image_esa/'
      config = {"image": image, "region":region, "scale":10, "crs":crs, "maxPixels":1e13, "description":file_name, "folder":folder, "fileNamePrefix":file_name, 'fileFormat': 'GeoTIFF'}
      task = ee.batch.Export.image.toDrive(**config)
      task.start()
      print(file_name)
      print('exporting')
      
def add_indices(image):
    nir = image.select('B8')
    swir_1 = image.select('B11')
    red = image.select('B4')
    green = image.select('B3')
    print(image.bandNames().getInfo())
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    mndwi = green.subtract(swir_1).divide(green.add(swir_1)).rename('MNDWI')
    ndbi =  swir_1.subtract(nir).divide(swir_1.add(nir)).rename('NDBI')
    scl = image.select('SCL')
    # print(image.dt)
    return image.addBands([ndvi,mndwi,ndbi,scl])


def get_masked_median(region, start_date, end_date, s2_all=s2):
  criteria = ee.Filter.And(ee.Filter.bounds(region), ee.Filter.date(start_date, end_date))
  s2 = s2_all.filter(criteria).map(maskEdges)
  

  print(ee.Element(s2.first()).get('MGRS_TILE').getInfo())
  chip = s2.first().get('MGRS_TILE').getInfo()
  zone = '7'
  if chip[2] >= 'N': zone = '6'
  crs = 'EPSG:32'+zone+chip[:2]
  print(crs, chip)
  srWithCloudMask = ee.Join.saveFirst('cloud_mask').apply(**{
        'primary': s2,
        'secondary': s2Clouds,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })}
        )

  return ee.ImageCollection(srWithCloudMask).map(maskClouds).median().clip(region), crs

def infer_collect(date, geom):

    start_date = ee.Date(date[0])
    end_date = ee.Date(date[1])
    now = datetime.datetime.utcnow()
    bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'NDVI', 'MNDWI', 'NDBI', 'SCL']
    dest = f"sentinel2_level2A_median_{now}"
    region = ee.Geometry.Rectangle(geom)
    median, crs = get_masked_median(region, start_date = start_date, end_date=end_date)
    median = add_indices(median).select(bands)
    print(crs)
    print(median.select('NDVI').getInfo())
    print(median.bandNames().getInfo())
    download(median, crs, dest+'_median', region)
