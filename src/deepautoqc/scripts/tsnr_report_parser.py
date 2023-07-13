import base64
import glob
import io
import re
import webbrowser
from pathlib import Path

import numpy as np
from data_structures import BrainScan
from PIL import Image
from script_utils import find_reports, get_user_input, parse_args, parse_svg
from utils import save_to_pickle


def process_image(image_path: str, save_path: str) -> None:
    axes_elements = parse_svg(image_path)
    results = []
    base_image_name = Path(image_path).name.split("_tsnr_rpt")[0]
    dataset_dir_match = re.search(r"(ds-[\w-]+)", image_path)
    if dataset_dir_match is not None:
        dataset_name = dataset_dir_match.group()
    else:
        dataset_name = ""

    for i, item in enumerate(axes_elements):
        (image_element,) = item.getElementsByTagName("image")
        xlink_ns = "http://www.w3.org/1999/xlink"
        base64_data = image_element.getAttributeNS(xlink_ns, "href")

        image_bytes = base64.b64decode(base64_data.split(",")[-1])
        image_obj = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(image_obj)

        row = (i // 7) + 1
        column = (i % 7) + 1
        result_id = f"{dataset_name}_{base_image_name}_report-tsnr_{row}-{column}"
        result = BrainScan(id=result_id, img=image_array, label="usable")
        results.append(result)

    if ARGS.user:
        webbrowser.open("file://" + image_path)
        user_input = get_user_input()
        if user_input:
            user_tuples = user_input.split(";")
            parsed_tuples = [tuple(map(int, t.split(","))) for t in user_tuples]
            unusable_positions = [(row - 1, col - 1) for row, col in parsed_tuples]
            for i, result in enumerate(results):
                row, col = i // 7, i % 7
                if (row, col) in unusable_positions:
                    results[i] = result._replace(label="unusable")

    pickle_filename = f"{dataset_name}_{base_image_name}_report-tsnr.pkl"
    pickle_path = Path(save_path) / pickle_filename
    save_to_pickle(data=results, file_path=pickle_path)


def find_tsnr_reports(base_path):
    base_path = Path(base_path).resolve()
    pattern = str(base_path / "**" / "*_tsnr_rpt.svg")
    report_paths = glob.glob(pattern, recursive=True)
    return report_paths


if __name__ == "__main__":
    ARGS = parse_args()
    print(f"Arguments: {ARGS}")
    report_type = "tsnr"
    report_paths = find_reports(ARGS.datapath, report_type=report_type)
    print(len(report_paths))
    for path in report_paths:
        try:
            process_image(image_path=path, save_path=ARGS.savepath)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
