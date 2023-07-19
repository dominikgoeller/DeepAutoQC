import argparse
import glob
import re
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog
from typing import List, Optional
from xml.dom.minidom import parse


def get_user_input() -> Optional[str]:
    root = tk.Tk()
    root.withdraw()
    pattern = r"^(\d+,\d+;)*(\d+,\d+)?$"
    while True:
        user_input = simpledialog.askstring(
            "Input",
            "Enter rows and columns of unusable images (format: row,column;row,column;...):",
        )
        root.destroy()
        if user_input is None:
            sys.exit(0)
        elif not user_input or re.fullmatch(pattern, user_input):
            return user_input
        messagebox.showerror(
            "Invalid input",
            "Please enter the rows and columns in the correct format (row,column;row,column;...)",
        )


def parse_svg(report_path: str):
    document = parse(report_path)
    g_elements = document.getElementsByTagName("g")
    axes_elements = [
        elem
        for elem in g_elements
        if "axes" in elem.getAttribute("id")
        and "axes_1" != elem.getAttribute("id")
        and "axes_9" != elem.getAttribute("id")
    ]
    return axes_elements


def find_reports(base_path: str, report_type: str) -> List[str]:
    base_path_obj = Path(base_path).resolve()

    if report_type == "tsnr":
        pattern = "*_tsnr_rpt.svg"
    elif report_type == "skull_strip":
        pattern = "*_skull_strip_report.svg"
    else:
        raise ValueError(f"Unknown report type: {report_type}")

    search_pattern = str(base_path_obj / "**" / pattern)

    report_paths = glob.glob(search_pattern, recursive=True)

    return report_paths


def parse_args():
    parser = argparse.ArgumentParser(description="SVG Parse Script")
    parser.add_argument(
        "-d",
        "--datapath",
        help="File Path to SVG",
    )
    parser.add_argument(
        "-s",
        "--savepath",
        help="Path to save pickle",
    )
    parser.add_argument(
        "-u",
        "--user",
        action="store_true",
        help="User review of images",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="usable",
        choices=["usable", "unusable"],
        help="Automatically sets the label to either usable or unusable",
    )
    args = parser.parse_args()

    return args
