import click





@click.group()
def main(prog_name="afrimap"):
    """
    Command to download, preprocess, train and run inference for landcover classification of Africa.
    """
    pass

@main.command()
@click.option("--type", required=True, type=click.Choice(
    [
        "train",
        "infer"
    ], case_sensitive=False,
), help="choose task type, train for training, infer for inference")
@click.option("--xmin", default=None, help=" xMin of xMin, yMin, xMax, yMax for area of interest", type=float)
@click.option("--xmax", default=None, help=" xMax of xMin, yMin, xMax, yMax for area of interest", type=float)
@click.option("--ymin", default=None, help=" yMin of xMin, yMin, xMax, yMax for area of interest", type=float)
@click.option("--ymax", default=None, help=" xMax of xMin, yMin, xMax, yMax for area of interest", type=float)
@click.option("--start_date", default=None, help="start date for filtering collection by date in YYYY-mm-dd format")
@click.option("--end_date", default=None, help="end date for filtering collection by date in YYYY-mm-dd format")
@click.option("--dest_name", default=None, help="name to include in destination file(for inference only)")
@click.option("--mlhub_key", default=None, help="MLHUB_API_KEY from Radient MLHUB(https://mlhub.earth/profile)")
def collect_data(type, xmin, xmax, ymin, ymax, start_date, end_date, dest_name, mlhub_key):
    if(type == "infer"):
        assert None not in [xmin, xmax, ymin, ymax], "geometry for area of interest is not provided."
        assert None not in [start_date, end_date], "value for start or end date is not provided " 
        from afrimap.data_collection.infer_collect import infer_collect 
        infer_collect(date=[start_date, end_date], geom=[xmin, ymin, xmax, ymax], dest_name=dest_name)
    else:
        assert mlhub_key, "please provide MLHUB_API_KEY after signing up for an account on Radiant MLHUB(https://mlhub.earth/profile)"
        from afrimap.data_collection.train_collect import train_collect
        train_collect(mlhub_key=mlhub_key)
        
        
@main.command()
@click.option("--type", required=True, type=click.Choice(
    [
        "train",
        "infer"
    ], case_sensitive=False,
), help="choose task type, train for training, infer for inference")
@click.option("--image_path", required= True, help=" path to the folder containing the satellite image for the dataset", type=str)
@click.option("--label_path", default=None, help=" path to the folder containing the ground truth or labels(needed only for training)", type=str)
def prep_data(type, image_path, label_path):
    if(type == "train"):
        assert label_path, "label path not provided"
        from afrimap.data_preparation.train_prep import train_prep
        train_prep(image_path=image_path, label_path=label_path)
    else:
        from afrimap.data_preparation.infer_prep import infer_prep
        infer_prep(image_path=image_path)

@main.command()
@click.option("--type", required=True, type=click.Choice(
    [
        "train",
        "infer"
    ], case_sensitive=False,
), help="choose task type, train for training, infer for inference")
@click.option("--image_path", required= True, help=" path to the folder containing the satellite image for the dataset", type=str)
@click.option("--label_path", default=None, help=" path to the folder containing the groundtruth or labels", type=str)
@click.option("--nb_epochs", default=10, help=" number of epoches for training", type=int)
@click.option("--lr", default=0.002, help=" learning rate", type=float)
@click.option("--model_path", default=None, help=" path to pre-trained model", type=str)
def train_infer(type, image_path, label_path, model_path, nb_epochs, lr):
    if(type == "train"):
        assert (None not in [image_path, label_path]), "Please provide both image_path and label_path"
        from afrimap.train_infer.train import train
        train(image_path=image_path, label_path=label_path, lr=lr, nb_epochs=nb_epochs, tr=True)
    else:
        assert None not in [image_path, model_path], "Please provide both image_path and model_path"
        from afrimap.train_infer.infer import infer
        infer(dataset_path=image_path, model_path=model_path)


@main.command()
@click.option("--predictions", required= True, help=" path to the folder containing the inferred tif files", type=str)
@click.option("--destination", default=None, help=" destination of the mosaic tiff file", type=str)
def post_process(predictions, destination):
    from afrimap.post_process.mosaic import mosaic
    mosaic(predictions=predictions, destination=destination)

if __name__ == "__main__":
    main(prog_name="afrimap")