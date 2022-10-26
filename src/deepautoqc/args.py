"""
Config Class to define hyperparameters more smoothly
"""


class config:
    """
    Dataset and Model
    """

    DATA_PATH = "./data/training_data"
    TEST_DATA_PATH = "./data/test_data"
    num_classes = 2
    SEED = 42
    requires_grad = True
    """
    Hyper Arguments
    """
    n_gpus = 4
    batch_size = 16
    epochs = 50
    num_workers = (
        32  # checked in jupyter notebook for optimal value. 2 epochs in 229 secs
    )
    lr = 1e-3
    momentum = 0.9
    optimizer = "SGD"  # or "ADAM"
    """
    Folders for checkpoints
    """
    EARLYSTOP_PATH = "./results/ResNet50"
    DUMMY_MODEL_CKPT = "./results/ResNet50/dummybestmodel.pt"
