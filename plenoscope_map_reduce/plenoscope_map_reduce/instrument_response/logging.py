import datetime
import json
import os


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


def reduce(
    list_of_log_paths,
    keys=[
        'corsika',
        'merlict',
        'grid',
        'export_grid_images',
        'prepare_trigger',
        'trigger',
        'cherenkov_classification',
        'feature_extraction',
    ]
):
    logs = []
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
            logs.append(run)
    return logs
