# Deep-AutoQC
Image classification for the ENIGMA HALFpipe project at Charité  using deep learning - BA Thesis
DeepAutoQC aims to assist researchers in classifying skull strip reports as either usable (good) or unusable (bad).

## Folder Structure
````
DeepAutoQC/
|
|--- src/
|   |--- DeepAutoQC.egg-info - package info
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

## Usage
The code in this project can be used by running either `train.py` script or `test.py` script within the command line.
First of all, you should go to `args.py` and
## For Training
* set the `DATA_PATH` constant to a folder only containing corresponding t1w.nii(.gz), mask.nii(.gz) files as this is required for a training run
* specify the hyperparameters used in training.
* set `EARLYSTOP_PATH` to your desired location where model_weights, optimizer_weights and more parameters will be saved

Then, run `python3 train.py` to start your training process. The progress is logged on the command line.

## Using multiple GPUs
You can enable multiple GPUs for training by setting `n_gpus` in `args.py`. If `n_gpus` is set to a value not available on your machine the function `device_preparation` will set `torch.device` to either `"cpu"` or `"cuda"` using all available GPUs.

## Resume Training
You can resume training from a previously saved checkpoint by:
`python train.py -r ckpt/path`

## For Predictions
* check the `ckpts/` folder and set `MODEL_CKPT` to your desired `*model.pt` which will be loaded for predictions

Then, run `python3 test.py [-h] -i INPUT`. Remember `the following arguments are required: -i/--input` should contain the path to your svg file.
Check `predictions` folder for your output.
