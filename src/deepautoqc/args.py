"""
Config Class to define hyperparameters more smoothly
"""


class config:
    """
    Dataset and Model
    """

    DATA_PATH = (
        "/data/gpfs-1/users/goellerd_c/work/usable"  # path to data on hpc-cluster
    )
    TEST_DATA_PATH = "./data/test_data"
    num_classes = 2
    SEED = 42
    requires_grad = True
    """
    Hyper Arguments
    """
    n_gpus = 4
    batch_size = 16
    epochs = 3
    num_workers = (
        32  # checked in jupyter notebook for optimal value. 2 epochs in 229 secs
    )
    lr = 1e-3
    momentum = 0.9
    optimizer = "SGD"  # or "ADAM"
    """
    Folders for checkpoints
    """
    EARLYSTOP_PATH = "/data/gpfs-1/users/goellerd_c/work/saved_models/ResNet50trueSGD.pt"  # path to save model on hpc-cluster
    DUMMY_MODEL_CKPT = "./results/ResNet50/EarlyStopcheckpoint2510.pt"  # src/deepautoqc/results/ResNet50/EarlyStopcheckpoint2510.pt
