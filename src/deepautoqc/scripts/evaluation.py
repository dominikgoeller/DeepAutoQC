# goal: zero false positive (bad classified as good) at how many false negatives (good classified as bad)
# we want to minimize the manual classification task to reduce number of datapoints researchers need to re-evaluate (all bad classified samples)

# we need: model, test_data in form of List[BrainScan]=label.good/bad, test_data contains bad and good samples (ratio is irrelevant?)
# accuracy can be neglected for our task, higher importance to confusion matrix and precision/recall curve


# https://en.wikipedia.org/wiki/Rule_of_three_(statistics)

import argparse
import pickle
import re

# load train_compr_unpacked report_paths
# x_array= ae_model.encoder(brainscan.img) for i in report_paths
# umap/tsn plotting von feature array
# y_array= brainscan.label for i in report_paths
# classifier = oneclassSVM
# classifier.fit(x_array)
# load test_compr_original List[BrainScan]
# results_dict{} key=namewithlabel value=prediction
# results_dict[test_data] = brainscan.label
# for each brainscan.img in List[BrainScan] if classifier.predict(brainscan.img) is outlier --> negative_pred otherwise positive_pred
# from results_dict build confusion_matrix
from pathlib import Path
from typing import Dict, Generator, List

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchio as tio
import umap.umap_ as umap
import wandb
import zstandard
from sklearn.metrics import confusion_matrix
from sklearn.svm import OneClassSVM

from deepautoqc.ae_arch2 import Autoencoder
from deepautoqc.data_structures import BrainScan


def load_data(data_path: str) -> Generator[Path, None, None]:
    """
    Load data paths from the specified directory.

    Parameters:
    - data_path (str): The file path to the data directory.

    Returns:
    - Generator[Path]: A generator object that yields paths to data files.
    """
    path = Path(data_path)
    return path.glob("*.zst")


def load_model(ckpt_path: str) -> Autoencoder:
    """
    Load the autoencoder model from a checkpoint.

    Parameters:
    - ckpt_path (str): The file path to the model checkpoint.

    Returns:
    - Autoencoder: The loaded autoencoder model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = Autoencoder().load_from_checkpoint(
            checkpoint_path=ckpt_path, map_location=device
        )
    except FileNotFoundError:
        print(f"Checkpoint file not found at {ckpt_path}.")
        raise
    return model


def single_brainscan_loader(scan_path):
    decompressor = zstandard.ZstdDecompressor()
    with open(scan_path, "rb") as compressed_file:
        # print("SCANPATH", scan_path)
        compressed_data = compressed_file.read()
    uncompressed_data = decompressor.decompress(compressed_data)
    item: BrainScan = pickle.loads(uncompressed_data)
    return item


def report_brainscan_loader(scan_path):
    decompressor = zstandard.ZstdDecompressor()
    with open(scan_path, "rb") as compressed_file:
        # print("SCANPATH", scan_path)
        compressed_data = compressed_file.read()
    uncompressed_data = decompressor.decompress(compressed_data)
    items: List[BrainScan] = pickle.loads(uncompressed_data)
    return items


def load_to_tensor(img: np.ndarray) -> torch.Tensor:
    transform = tio.CropOrPad((3, 704, 800))
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img)
    img = tio.ScalarImage(tensor=img[None])
    img = transform(img)
    img = img.data[0]

    return img.float()


def build_feature_matrix(model: Autoencoder, datapoints_generator):
    X = []
    i = 0
    for path in datapoints_generator:
        if i > 25000:
            break
        brainscan_obj: BrainScan = single_brainscan_loader(scan_path=path)
        img_tensor = load_to_tensor(img=brainscan_obj.img)
        img_tensor = img_tensor.to(model.device).unsqueeze(0)
        with torch.no_grad():
            feature_vector = model.encoder(img_tensor)
            feature_vector = feature_vector.squeeze(0)
            X.append(feature_vector.cpu().numpy())
        i += 1
    X_array = np.array(X)
    print(X_array.shape)
    return X_array


def fit_classifier(feature_matrix):
    clf = OneClassSVM().fit(feature_matrix)

    return clf


def build_test_data_matrix(test_path_generator, model: Autoencoder):
    test_dict: Dict = {}  # key=filename, value=List of img encoded vectors
    for path in test_path_generator:
        path_name = path.name
        brainscan_list: List[BrainScan] = report_brainscan_loader(scan_path=path)
        if path.name not in test_dict:
            test_dict[path.name] = []
        for brainscan_obj in brainscan_list:
            img_tensor = load_to_tensor(img=brainscan_obj.img)
            img_tensor = img_tensor.to(model.device).unsqueeze(0)
            with torch.no_grad():
                feature_vector = model.encoder(img_tensor)
                feature_vector = feature_vector.squeeze(0)
                test_dict[path_name].append(feature_vector.cpu().numpy())
    return test_dict


def make_predictions(clf, test_dict):
    """
    Make predictions for each report in the test dictionary using the provided classifier.
    Each report is composed of several data points, and the report is considered an outlier
    if any of these data points is classified as an outlier by the classifier.

    Parameters:
    - clf: The trained classifier with a predict method.
    - test_dict (dict): A dictionary with keys as report names and values as lists of data for prediction.

    Returns:
    - results_dict (dict): A dictionary with keys as report names and values as the prediction results.
    """
    results_dict = {}

    for report_name, report_data in test_dict.items():
        x_matrix = np.array(report_data)
        print(x_matrix.shape)
        if x_matrix.shape[1] != 64:
            raise ValueError(
                f"Data for report {report_name} is not in the expected shape (21, 64)"
            )
        predictions = clf.predict(x_matrix)
        results_dict[report_name] = -1 if -1 in predictions else 1

    return results_dict


def calculate_conf_matrix(results_dict: Dict[str, int]) -> None:
    """
    Calculate and print the confusion matrix from the prediction results.

    Parameters:
    - results_dict (dict): A dictionary with keys as names with labels and values as predictions.
    """
    y_true = []
    y_pred = []
    for k, v in results_dict.items():
        y_true.append(-1 if "label-bad" in k else 1)
        y_pred.append(v)
    CM = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print(CM)
    _ = CM[0][0]  # TN bad as bad
    FN = CM[1][0]
    _ = CM[1][1]  # TP good as good
    FP = CM[0][1]
    print(f"FP (bad as good): {FP} versus FN (good as bad): {FN}")


def visualize_features(features):
    wandb.init(project="Feature Visualization", entity="dominikgoeller")

    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(features)
    labels = None

    plt.figure(figsize=(10, 8))
    plot = sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=labels,
        palette=sns.color_palette("hsv", None),
        legend="full",
    )
    plt.title("UMAP projection of the Features")
    # plot_filename = ""
    # plt.savefig(plot_filename)

    wandb.log({"UMAP Visualization": wandb.Image(plot)})

    wandb.finish()


def build_feature_dict(model: Autoencoder, datapoints_generator):
    # X = []
    feat_dict = {}
    # i = 0
    for path in datapoints_generator:
        # if i > 2000:
        # break
        brainscan_obj: BrainScan = single_brainscan_loader(scan_path=path)
        img_tensor = load_to_tensor(img=brainscan_obj.img)
        img_tensor = img_tensor.to(model.device).unsqueeze(0)
        with torch.no_grad():
            feature_vector = model.encoder(img_tensor).squeeze(0).cpu()
            # X.append(feature_vector.cpu().numpy())
            feat_dict[brainscan_obj.id] = feature_vector.numpy()
        # i += 1
    # X_array = np.array(X)
    # print(X_array.shape)
    # return X_array
    compressor = zstandard.ZstdCompressor()
    compressed_data = compressor.compress(pickle.dumps(feat_dict))
    with open(
        Path(
            "/data/gpfs-1/users/goellerd_c/work/deep-auto-qc/parsed_dataset/skull_strip_report/ae_data/"
        ).joinpath("feature_dict.pkl.zst"),
        "wb",
    ) as compressed_file:
        compressed_file.write(compressed_data)
    return feat_dict


def visualize_features_dict(feat_dict):
    wandb.init(project="Feature Visualization", entity="dominikgoeller")
    features = np.array(list(feat_dict.values()))
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(features)
    labels_set = set(extract_ds(key) for key in feat_dict.keys())
    labels = [extract_ds(key) for key in feat_dict.keys()]
    # print(embedding.shape)
    n_labels = len(labels_set)
    palette = sns.color_palette(cc.glasbey, n_colors=n_labels)
    # sns.color_palette("hsv", None)
    plt.figure(figsize=(10, 8))
    plot = sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=labels,
        palette=palette,
        legend="full",
    )
    plt.title("UMAP projection of the Features")
    # plot_filename = ""
    # plt.savefig(plot_filename)

    wandb.log({"UMAP Visualization": wandb.Image(plot)})

    wandb.finish()


def extract_ds(key):
    # This pattern looks for 'ds-' followed by any characters until the next underscore
    match = re.search(r"(ds-[a-z0-9]+)_", key)
    return match.group(1) if match else None


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation Script path inputs")
    parser.add_argument(
        "--train",
        help="File Path to training data",
    )
    parser.add_argument(
        "-c",
        "--ckptpath",
        help="Path to model checkpoint",
    )
    parser.add_argument("--test", default=None, help="File path to test data")

    args = parser.parse_args()

    return args


def main():
    ARGS = parse_args()
    train_path = ARGS.train
    train_data_gen = load_data(data_path=train_path)

    ckpt_path = ARGS.ckptpath
    model = load_model(ckpt_path=ckpt_path)

    # X_array = build_feature_matrix(model=model, datapoints_generator=train_data_gen)
    feat_dict = build_feature_dict(model=model, datapoints_generator=train_data_gen)

    visualize_features_dict(feat_dict=feat_dict)

    # clf = fit_classifier(feature_matrix=X_array)

    # test_path = ARGS.test
    # test_path_gen = load_data(data_path=test_path)

    # test_dict = build_test_data_matrix(test_path_generator=test_path_gen, model=model)

    # results_dict = make_predictions(clf=clf, test_dict=test_dict)

    # calculate_conf_matrix(results_dict=results_dict)


if __name__ == "__main__":
    main()
