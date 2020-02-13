import numpy as np
import pandas as pd
import msgpack


KEYS = {'a': '<i8', 'b': '<f8'}


list_of_dicts = [
    {'a': int(1), 'b': float(2)},
    {'a': int(2), 'b': float(1)},
    {'a': int(3), 'b': float(0)},
]

rec = pd.DataFrame(list_of_dicts).to_records(index=False)

msg = msgpack.dumps(pd.DataFrame(rec).to_dict(orient='lists'))

dict_back = msgpack.loads(msg, raw=False)
rec_back = pd.DataFrame(dict_back).to_records(index=False)


def empty_recarray(keys):
    dtypes = []
    for k in keys:
        dtypes.append((k, keys[k]))
    return np.recarray(
        shape=(0, len(keys)),
        dtype=dtypes)

def empty_DataFrame(keys):


er = empty_recarray(keys=KEYS)

emsg = msgpack.dumps(pd.DataFrame(er).to_dict(orient='lists'))