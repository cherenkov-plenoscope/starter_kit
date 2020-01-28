import datetime
import json


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
