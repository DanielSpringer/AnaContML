import os
from pathlib import Path
import stat

def set_permissions(outpath: Path):

    full_path = outpath.resolve()
    # Only set permissions when writing to public
    if "iarai/public" not in str(full_path):
        return
    os.chmod(full_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    # set permission for files
    for f in full_path.glob("*"):
        os.chmod(f, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    # set permission for parents recursively
    for d in full_path.parents:
        if "w_max_" in str(d) and "w_steps" in str(d):
            os.chmod(d, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
