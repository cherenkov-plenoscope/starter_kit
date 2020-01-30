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
                "t": now.strftime("%Y-%m-%d_%H:%M:%S"),
                "delta_t": (now - self.last_log_time).total_seconds(),
                "msg": msg}
            f.write(json.dumps(d)+"\n")
        self.last_log_time = now


def reduce(
    list_of_log_paths,
    keys=[
        'run CORSIKA',
        'run merlict',
        'reuse, grid',
        'run sum-trigger',
        'prepare refocus-sum-trigger',
        'Cherenkov classification',
        'extract features from light-field',
        'export, level 1, and level 2',
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
                    run[logline['msg']] = logline['delta_t']
            logs.append(run)
    return logs
