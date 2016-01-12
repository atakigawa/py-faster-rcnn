"""Set up paths."""

import os.path as osp
import sys


def _add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def add_paths():
    this_dir = osp.dirname(__file__)

    # Add lib to PYTHONPATH
    lib_path = osp.join(this_dir, '..', '..')
    _add_path(lib_path)
