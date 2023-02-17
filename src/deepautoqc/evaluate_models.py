import argparse
import torch
from train import evaluate_model
from utils import device_preparation, load_from_pickle, load_model_new
from args import config
from data import TestSkullstripDataset, generate_test_loader

def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument(
        "-m",
        "--modelpath",
        default=None,
        type=str,
        help="path to trained model",
    )
    args = parser.parse_args()

    return args

def main(modelpath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data = load_from_pickle(
        "/data/gpfs-1/users/goellerd_c/work/T2_real_testset"
    )
    test_dataset = TestSkullstripDataset(test_data)
    test_loader = generate_test_loader(
        dataset=test_dataset,
        batchsize=1,
        num_workers=config.num_workers,
    )

    model = load_model_new(model_filepath=modelpath)

    evaluate_model(trained_model=model, test_loader=test_loader, device=device)


if __name__ == "__main__":
    ARGS = parse_args()
    main(
        modelpath=ARGS.modelpath
    )