import datetime
import time
import json
import os
import pandas as pd
import shutil


class JsonlLog:
    def __init__(self, path):
        self.path = path
        self.log("start")

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


KEYS = [
    "corsika",
    "merlict",
    "grid",
    "prepare_trigger",
    "trigger",
    "cherenkov_classification",
    "feature_extraction",
    "past_trigger_gz_tar",
    "export_event_table",
    "export_grid_histograms",
    "export_past_trigger",
]


def reduce(
    list_of_log_paths, out_path, keys=KEYS,
):
    log_records = reduce_into_records(
        list_of_log_paths=list_of_log_paths, keys=keys
    )
    log_df = pd.DataFrame(log_records)
    log_df.to_csv(out_path + ".tmp", index=False)
    shutil.move(out_path + ".tmp", out_path)


def reduce_into_records(list_of_log_paths, keys=KEYS):
    list_of_log_records = []
    for log_path in list_of_log_paths:
        run_id = int(os.path.basename(log_path)[0:6])
        run = {"run_id": run_id}
        for key in keys:
            run[key] = None
        with open(log_path, "rt") as fin:
            for line in fin:
                logline = json.loads(line)
                if logline["msg"] in keys:
                    run[logline["msg"]] = logline["delta"]
            list_of_log_records.append(run)
    return list_of_log_records
