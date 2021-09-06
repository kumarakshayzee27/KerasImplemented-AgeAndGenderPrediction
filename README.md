# Age and Gender Estimation
This is a Keras implementation of a CNN for estimating age and gender from a face image

## Dependencies
- Python3.6+


## Usage

### Use trained model for demo
Run the demo script (requires web cam).
You can use `--image_dir [IMAGE_DIR]` option to use images in the `[IMAGE_DIR]` directory instead.

```sh
python demo.py
```

The trained model will be automatically downloaded to the `pretrained_models` directory.

### Create training data from the IMDB-WIKI dataset
First, download the dataset.
The dataset is downloaded and extracted to the `data` directory by:

```sh
./download.sh
```

Secondly, filter out noise data and serialize labels into `.csv` file.
Please check [check_dataset.ipynb](check_dataset.ipynb) for the details of the dataset.
The training data is created by:

```sh
python create_db.py --db imdb
```

```sh
usage: create_db.py [-h] [--db DB] [--min_score MIN_SCORE]

This script cleans-up noisy labels and creates database for training.


### Create training data from the UTKFace dataset [currently not supported]
Firstly, download images from [the website of the UTKFace dataset](https://susanqq.github.io/UTKFace/).
`UTKFace.tar.gz` can be downloaded from `Aligned&Cropped Faces` in Datasets section.
Then, extract the archive.

```sh
tar zxf UTKFace.tar.gz UTKFace
```

Finally, run the following script to create the training data:

```
python create_db_utkface.py -i UTKFace -o UTKFace.mat
```


```sh
python demo.py --weight_file WEIGHT_FILE --margin 0
```

```sh
python train.py
```

Trained weight files are stored as `checkpoints/*.hdf5` for each epoch if the validation loss becomes minimum over previous epochs.

#### Changing model or the other training parameters
You can change [default setting(s)](src/config.yaml) from command line as:

```sh
python train.py model.model_name=EfficientNetB3 model.batch_size=64
```

### Use the trained model

```sh
python demo.py
```

```sh
usage: demo.py [-h] [--weight_file WEIGHT_FILE] [--margin MARGIN]
               [--image_dir IMAGE_DIR]

This script detects faces from web cam input, and estimates age and gender for
the detected faces.

Please use the best model among `checkpoints/*.hdf5` for `WEIGHT_FILE` if you use your own trained models.



### Estimated results
Trained on imdb, tested on wiki.


### Evaluation

#### Evaluation on the APPA-REAL dataset
You can evaluate a trained model on the APPA-REAL (validation) dataset by:

```bash
python evaluate_appa_real.py --weight_file WEIGHT_FILE
```


