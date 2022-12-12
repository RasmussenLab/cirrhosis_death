import logging
import pathlib

logger = logging.getLogger(__name__)


def add_suffix(fpath: pathlib.Path, suffix: str) -> pathlib.Path:
    """Add suffix to name of a filepath."""
    fname = fpath.parent / (fpath.stem + suffix + fpath.suffix)
    logger.info("New filename:", fname)
    return fname


def print_files(files: dict[str, pathlib.Path]):
    """Print files for snakemake rule."""
    print(',\n'.join(f'{k}="{v.as_posix()}"' for k, v in files.items()))
