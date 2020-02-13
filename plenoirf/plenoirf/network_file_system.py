import uuid
import os
import shutil
import errno


def copy(src, dst):
    """
    Atomic copy.
    """
    copy_id = uuid.uuid4().__str__()
    tmp_dst = "{:s}.{:s}.tmp".format(dst, copy_id)
    try:
        shutil.copytree(src, tmp_dst)
    except OSError as exc:
        if exc.errno == errno.ENOTDIR:
            shutil.copy2(src, tmp_dst)
        else:
            raise
    os.rename(tmp_dst, dst)


def move(src, dst):
    """
    Atomic move across seperate file-system.
    """
    try:
        os.rename(src, dst)
    except OSError as err:
        if err.errno == errno.EXDEV:
            copy(src, dst)
            os.unlink(src)
        else:
            raise
