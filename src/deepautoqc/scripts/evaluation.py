# goal: zero false negatives (bad classified as good) at how many false positives (good classified as bad)
# we want to minimize the manual classification task to reduce number of datapoints researchers need to re-evaluate (all bad classified samples)

# we need: model, test_data in form of List[BrainScan]=label.good/bad, test_data contains bad and good samples (ratio is irrelevant?)
# accuracy can be neglected for our task, higher importance to confusion matrix and precision/recall curve


# https://en.wikipedia.org/wiki/Rule_of_three_(statistics)

import argparse
import pickle

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
from typing import Dict, List

import numpy as np
import torch
import torchio as tio
import zstandard
from sklearn.metrics import confusion_matrix
from sklearn.svm import OneClassSVM

from deepautoqc.ae_arch2 import Autoencoder
from deepautoqc.data_structures import BrainScan


def load_train_data(train_path):
    """Returns generator object of train data paths"""
    train_path = Path(train_path)
    train_path_generator = train_path.glob("*.zst")
    return train_path_generator


def load_test_data(test_path):
    test_path = Path(test_path)
    test_path_generator = test_path.glob("*.zst")
    return test_path_generator


def load_model(ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Autoencoder().load_from_checkpoint(
        checkpoint_path=ckpt_path,
        map_location=device,
    )
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
        if i > 1000:
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
        for brainscan_obj in brainscan_list:
            img_tensor = load_to_tensor(img=brainscan_obj.img)
            img_tensor = img_tensor.to(model.device).unsqueeze(0)
            with torch.no_grad():
                feature_vector = model.encoder(img_tensor)
                feature_vector = feature_vector.squeeze(0)
                test_dict[path_name].append(feature_vector.cpu().numpy())
    return test_dict


def make_predictions(clf, test_dict):
    results_dict = {}  # key = name containing label, value = prediction
    # x_test_array = np.array(list(test_dict.values()))
    # x_test_names = list(test_dict.keys())
    # preds = clf.predict(x_test_array)  # one class svm outputs +1 inliers or -1 outliers
    for k, v in test_dict.items():
        x_array = np.array(v)
        print("array shape of one report", x_array.shape)
        preds = clf.predict(x_array)
        if -1 in preds:  # outlier detected inside this report!
            print("predictions for this report", preds)
            results_dict[k] = -1
        results_dict[k] = 1  # no outliers detected
    return results_dict


def calculate_conf_matrix(results_dict: dict):
    y_true = []
    y_pred = []
    for k, v in results_dict.items():
        if "label-bad" in k:
            y_true.append(-1)  # outlier
        else:
            y_true.append(1)  # inlier = label good
        y_pred.append(v)
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    print(conf_matrix)


def visualize_features():
    # TODO
    pass


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
    parser.add_argument("--test", help="File path to test data")

    args = parser.parse_args()

    return args


def main():
    ARGS = parse_args()
    train_path = ARGS.train
    train_data_gen = load_train_data(train_path=train_path)

    ckpt_path = ARGS.ckptpath
    model = load_model(ckpt_path=ckpt_path)

    X_array = build_feature_matrix(model=model, datapoints_generator=train_data_gen)

    clf = fit_classifier(feature_matrix=X_array)

    test_path = ARGS.test
    test_path_gen = load_test_data(test_path=test_path)

    test_dict = build_test_data_matrix(test_path_generator=test_path_gen, model=model)

    results_dict = make_predictions(clf=clf, test_dict=test_dict)

    calculate_conf_matrix(results_dict=results_dict)


if __name__ == "__main__":
    main()
