from pathlib import Path

"""
Config Class to define hyperparameters more smoothly
"""


class config:
    """
    Dataset and Model
    """

    DATA_PATH = Path(
        "/data/gpfs-1/users/goellerd_c/work/usable"  # path to data on hpc-cluster
    )
    # TEST_DATA_PATH = "./data/test_data"
    num_classes = 2
    SEED = 42
    requires_grad = True
    """
    Hyper Arguments
    """
    n_gpus = 4
    batch_size = 16
    epochs = 30
    num_workers = 32  # checked in jupyter notebook for optimal value. 2 training (w/o validation) epochs in 229 secs
    lr = 1e-3
    momentum = 0.9
    optimizer = "SGD"  # or "ADAM"
    patience = 7  # patience for early stopping
    """
    Paths for checkpoints
    """
    EARLYSTOP_PATH = "./ckpts/ResNet50/"  # path to save model
    MODEL_CKPT = "/Users/dominik/Downloads/ResNet50trueADAM.pt"  # Path to checkpoint to resume training or predict./results/ResNet50/EarlyStopcheckpoint2510.pt"  # src/deepautoqc/results/ResNet50/EarlyStopcheckpoint2510.pt
