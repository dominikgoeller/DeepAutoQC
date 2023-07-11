# Deep-AutoQC
Image classification for the ENIGMA HALFpipe project at Charité  using deep learning - BA Thesis
DeepAutoQC aims to assist researchers in classifying skull strip reports as either usable (good) or unusable (bad).

## Folder Structure
````
DeepAutoQC/
|
|--- src/
|   |--- deepautoqc/
|
|       |--- ckpts/ - folder containing checkpoints of saved models
|           |--- ResNet50/
|               | %Y-%m-%d/ - creates folder for each saved model in this time format
|                   |--- dummybestmodel.pt
|
|       |--- data/
|           |--- test_data/ - folder containing images for prediction
|               |--- dummytest.svg
|
|           |--- training_data/ all t1w, mask .nii files for the training process
|               |--- dummy_mask.nii.gz
|               |--- dummy_t1w.nii.gz
|
|       |--- predictions/ - outputfolder for predictions in json format
|           |--- sub_102008_skull_strip_report.json - example output file
|       |--- weights/
|           |--- S_Small-CBR.pt - example weight file of a trained model
|
|       |--- args.py - config class for training configuration
|       |--- data.py - custom dataset and dataloaders
|       |--- metrics.py - evaluation functions of training process
|       |--- models.py - model builder
|       |--- ni_image_helpers.py - performance boosts
|       |--- test.py - script to use model for predictions
|       |--- train.py - script to start training process
|       |--- utils.py - small utility functions
| ...
| - packaging files and pre-commit configurations
````

# Training Process

This script is designed for training the MRI Quality Control (MRIAutoQC) model on brain scan data. The main parts of this process are argument parsing, data preparation, model training, and testing.

## Argument Parsing

The script accepts several arguments that allow you to customize the training process:

- `-dl` or `--data_location`: Choose between `local` and `cluster` to determine the data paths. This determines whether the script uses data stored locally or on a cluster.
- `-mn` or `--model_name`: Selects the architecture of the Transfusion model used for training. Options include `small`, `tiny`, `wide`, and `tall`.
- `-e` or `--epochs`: The number of epochs to train the network. The default is 20.

## Data Preparation

The data for the model is prepared using the `BrainScanDataModule`. This PyTorch Lightning DataModule is responsible for loading the brain scan data from either a local path or a cluster, and preparing it for training.

## Model Training

The `MRIAutoQC` model is trained using the PyTorch Lightning Trainer. The trainer is set up with several options:

- `accelerator="auto"`: Automatically selects the appropriate hardware accelerator (CPU, GPU, or TPU) available on the machine.
- `deterministic="warn"`: If any operations are performed that could cause non-deterministic behavior, a warning will be raised.
- `enable_progress_bar=True`: Enables a progress bar to be displayed during training.
- `max_epochs=args.epochs`: Sets the maximum number of epochs for training.

The trainer is then used to fit the model on the training and validation data, which are obtained from the DataModule.

## Testing

After training, the model is tested on the test data, which is again obtained from the DataModule.

## Running the Script

To run the script, you can use a command similar to the following:

```
python train.py --data_location local --model_name small --epochs 30
```

This command would train the `small` model on `local` data for `30` epochs. Adjust the arguments as needed for your specific use case.

~~## Resume Training~~
~~You can resume training from a previously saved checkpoint by:~~
~~`python train.py -r ckpt/path`~~

~~## For Predictions~~
~~* check the `ckpts/` folder and set `MODEL_CKPT` to your desired `*model.pt` which will be loaded for predictions~~

~~Then, run `python3 test.py [-h] -i INPUT`. Remember `the following arguments are required: -i/--input` should contain the path to your svg file.~~
~~Check `predictions` folder for your output.~~

~~## Weights~~
~~* Weights of every trained model are stored in `/src/deepautoqc/weights` with their respective names and the dataset on which they got trained on~~
