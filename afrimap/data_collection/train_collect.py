  
from radiant_mlhub import client
import os
import tarfile
from pathlib import Path


def train_collect(mlhub_key):
    outdir = 'afrimap/data_collection/train_data'
    os.environ['MLHUB_API_KEY'] = mlhub_key

    collection_ids = ['ref_landcovernet_af_v1_labels', 'ref_landcovernet_af_v1_source_sentinel_2' ]
    for collection_id in collection_ids:
        print(f'Downloading {collection_id}')
        zip_path= client.download_collection_archive(collection_id, output_dir=outdir, if_exists='resume')

        with tarfile.open(zip_path, 'r:gz') as tar:
            tar.extractall(Path(outdir))
      
    