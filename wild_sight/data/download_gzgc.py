#!/usr/bin/env python3
"""Downloads a dataset of giraffes and zebras.
Remember where you save the data because this will be needed in the training
configuration file."""

import argparse
import pathlib
import shutil
import tarfile
import tempfile
import requests


def download_data(save_dir: pathlib.Path):
    url = "https://lilablobssc.blob.core.windows.net/giraffe-zebra-id/gzgc.coco.tar.gz"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file = pathlib.Path(tmp_dir) / "gzgc.coco.tar.gz"

            with open(tmp_file, "wb") as f:
                shutil.copyfileobj(r.raw, f)

            with tarfile.open(tmp_file, "r:gz") as data:
                
                import os
                
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(data, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        required=True,
        help="Where to save the extracted data.",
    )
    args = parser.parse_args()

    download_data(args.save_dir.expanduser())
