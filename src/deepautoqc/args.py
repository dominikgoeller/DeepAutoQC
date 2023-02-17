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
        # "/mnt/mbServerProjects/HOME/USERS/lea/auto_qc/dataset/usable"
    )
    # TEST_DATA_PATH = "./data/test_data"
    num_classes = 2
    SEED = 42
    requires_grad = False
    """
    Hyper Arguments
    """
    n_gpus = 4
    batch_size = 32
    epochs = 50
    num_workers = 0  # checked in jupyter notebook for optimal value. 2 training (w/o validation) epochs in 229 secs
    lr = 1e-3
    momentum = 0.9
    optimizer = "ADAN"  # "ADAM", "SGD", "ADAN"
    patience = 7  # patience for early stopping
    """
    Paths for checkpoints
    """
    EARLYSTOP_PATH = "./ckpts/ResNet50/"  # path to save model
    # MODEL_CKPT = "/data/gpfs-1/users/goellerd_c/work/git_repos/DeepAutoQC/src/deepautoqc/ckpts/ResNet50/2022-12-05/ResNet_trainable_Adam.pt"  # Path to checkpoint to resume training or predict./results/ResNet50/EarlyStopcheckpoint2510.pt"  # src/deepautoqc/results/ResNet50/EarlyStopcheckpoint2510.pt
    MODEL_CKPT = "/data/gpfs-1/users/goellerd_c/work/git_repos/DeepAutoQC/src/deepautoqc/ckpts/ResNet50/2023-01-30/L_ResNet50_finetune_Adan.pt"  # "/Users/dominik/Downloads/ResNet_trainable_Adam.pt"
