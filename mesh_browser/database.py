

import os

from config import MESH_EXTENSIONS


def is_valid_database(path):
    """True if the path exists and is a folder."""
    return os.path.isdir(path)


def list_subfolders(database_dir):
    """Names of all subfolders in the database, sorted alphabetically."""
    return sorted(
        name for name in os.listdir(database_dir)
        if os.path.isdir(os.path.join(database_dir, name))
    )


def list_meshes(folder):
    """Names of all mesh files in a folder, sorted alphabetically."""
    return sorted(
        name for name in os.listdir(folder)
        if name.lower().endswith(MESH_EXTENSIONS)
    )