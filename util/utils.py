import torch
import os

def to_array(feature_map):
    if feature_map.shape[0] == 1:
        feature_map = feature_map.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    else:
        feature_map = feature_map.permute(0, 2, 3, 1).detach().cpu().numpy()
    return feature_map


def to_tensor(feature_map):
    return torch.as_tensor(feature_map.transpose(0, 3, 1, 2), dtype=torch.float32)


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)


def get_files(file_dir, ext=None):
    """
    get all file paths (full paths) in a directory with specific extension
    Args:
        file_dir: path to a directory
        ext: desired extension (e.g: '.jpg'). If none, returns any file
    Returns:
        list of full file paths
    """
    if not os.path.exists(file_dir):
        return []
    files = os.listdir(file_dir)
    paths = []
    for x in files:
        if ext is None:
            paths.append(os.path.join(file_dir, x))
        if isinstance(ext, str) and x.lower().endswith(ext):
            paths.append(os.path.join(file_dir, x))
        if isinstance(ext, list):
            for curr_ext in ext:
                if x.lower().endswith(curr_ext):
                    paths.append(os.path.join(file_dir, x))

    return paths


def get_files_recursive(file_dir, ext=None):
    """
    runs get_files on all subdirectories recursively and returns a concatenated file list
    Args:
        file_dir: path to a directory
        ext: desired extension (e.g: '.jpg'). If none, returns any file
    Returns:
        list of full file paths
    """
    if not os.path.exists(file_dir):
        return []
    files = get_files(file_dir, ext)
    entries = os.listdir(file_dir)
    for entry in entries:
        entry_path = os.path.join(file_dir, entry)
        if os.path.isdir(entry_path):
            files += get_files_recursive(entry_path, ext)
    return files
