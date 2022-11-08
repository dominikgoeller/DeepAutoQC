# Deep-AutoQC
Image classification for the ENIGMA HALFpipe project using deep learning - BA Thesis

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
|               |--- dummybestmodel.pt
|
|       |--- data/
|           |--- test_data/ - folder containing images for prediction
|               |--- dummytest.svg
|
|           |--- training_data/ folder containing all t1w, mask .nii files for the training process
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
