import numpy as np
import json


class Encoder(json.JSONEncoder):
    """
    json encoder for numpy types
    Thanks to:
        github-user: schouldsee, 'Bridging Bio and informatics'
        stackoverflow-user: tsveti_iko

    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def write(path, out_dict, indent=4):
    with open(path, "wt") as f:
        f.write(json.dumps(out_dict, indent=indent, cls=Encoder))


def read(path):
    with open(path, "rt") as f:
        out_dict = json.loads(f.read())
    return out_dict