import os
import kagglehub


def download_dataset(handle: str = "paultimothymooney/chest-xray-pneumonia") -> str:
    """Download a Kaggle dataset to a local ``datasets`` directory.

    The KaggleHub helper normally caches downloads under ``~/.cache/kagglehub``.
    This wrapper forces the files to land inside the repository so that they can
    be inspected, committed, or packaged alongside your code.

    Returns the path to the downloaded files (usually a directory or tarball).
    """

    # make an explicit relative folder inside the repo
    out_dir = os.path.join(os.getcwd(), "datasets")
    os.makedirs(out_dir, exist_ok=True)

    print(f"downloading dataset '{handle}' to {out_dir} ...")
    result_path = kagglehub.dataset_download(handle, output_dir=out_dir)

    print(f"download complete: {result_path}")
    return result_path


if __name__ == "__main__":
    download_dataset()
