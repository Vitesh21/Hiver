import os


def project_root_from_here(this_file: str) -> str:
    return os.path.dirname(os.path.dirname(this_file))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)