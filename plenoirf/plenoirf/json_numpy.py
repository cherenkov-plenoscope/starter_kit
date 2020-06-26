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
