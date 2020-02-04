import datetime
import json
import os
import pandas as pd
import shutil


class JsonlLog:
    def __init__(self, path):
        self.last_log_time = datetime.datetime.now()
        self.path = path
        self.log("start")

    def log(self, msg):
        now = datetime.datetime.now()
        with open(self.path, "at") as f:
            d = {
                "time": now.strftime("%Y-%m-%d %H:%M:%S"),
                "runtime": (now - self.last_log_time).total_seconds(),
                "msg": msg}
            f.write(json.dumps(d)+"\n")
        self.last_log_time = now

KEYS = [
    'corsika',
    'merlict',
    'grid',
    'prepare_trigger',
    'trigger',
    'cherenkov_classification',
    'feature_extraction',
    'past_trigger_gz_tar',
    'export_event_table',
    'export_grid_histograms',
    'export_past_trigger',
]


def reduce(
    list_of_log_paths,
    out_path,
    keys=KEYS,
):
    log_records = reduce_into_records(
        list_of_log_paths=list_of_log_paths,
        keys=keys)
    log_df = pd.DataFrame(log_records)
    log_df.to_csv(out_path+".tmp", index=False)
    shutil.move(out_path+".tmp", out_path)


def reduce_into_records(
    list_of_log_paths,
    keys=KEYS
):
    list_of_log_records = []
    for log_path in list_of_log_paths:
        run_id = int(os.path.basename(log_path)[0:6])
        run = {'run_id': run_id}
        for key in keys:
            run[key] = None
        with open(log_path, 'rt') as fin:
            for line in fin:
                logline = json.loads(line)
                if logline['msg'] in keys:
                    run[logline['msg']] = logline['runtime']
            list_of_log_records.append(run)
    return list_of_log_records
