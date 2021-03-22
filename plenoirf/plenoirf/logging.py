import datetime
import time
import json
import os
import pandas as pd
import shutil


class JsonlLog:
    def __init__(self, path):
        self.path = path

    def log(self, msg, delta=None):
        unix_time_now = time.time()
        date_now = datetime.datetime.fromtimestamp(unix_time_now)
        with open(self.path, "at") as f:
            d = {
                "time": date_now.isoformat(),
                "unix": unix_time_now,
                "msg": msg,
            }
            if delta:
                d["delta"] = delta
            f.write(json.dumps(d) + "\n")


class TimeDelta:
    def __init__(self, log, msg):
        self.log = log
        self.msg = msg

    def __enter__(self):
        self.start = time.time()
        self.log.log(msg=self.msg + ":start")
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop = time.time()
        self.log.log(msg=self.msg + ":stop", delta=self.delta())

    def delta(self):
        return self.stop - self.start


def reduce(list_of_log_paths, out_path):
    log_records = reduce_into_records(list_of_log_paths=list_of_log_paths)
    log_df = pd.DataFrame(log_records)
    log_df = log_df.set_index("run_id")
    log_df = log_df.sort_index()
    log_df.to_csv(out_path + ".tmp", index=False, na_rep="nan")
    shutil.move(out_path + ".tmp", out_path)


def reduce_into_records(list_of_log_paths):
    list_of_log_records = []
    for log_path in list_of_log_paths:
        run_id = int(os.path.basename(log_path)[0:6])
        run = {"run_id": run_id}

        with open(log_path, "rt") as fin:
            for line in fin:
                logline = json.loads(line)
                if "delta" in logline:
                    name = logline["msg"].replace(":stop", "")
                    run[name] = logline["delta"]
            list_of_log_records.append(run)

    return list_of_log_records
