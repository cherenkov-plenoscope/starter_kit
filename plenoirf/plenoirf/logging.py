"""
A simple json-line-logger
"""

import time
import json_numpy
import os
import pandas as pd
import shutil
import logging
import sys


DATEFMT_ISO8601 = "%Y-%m-%dT%H:%M:%S"
FMT = "{"
FMT += '"t":"%(asctime)s.%(msecs)03d"'
FMT += ", "
FMT += '"c":"%(module)s:%(funcName)s"'
FMT += ", "
FMT += '"l":"%(levelname)s"'
FMT += ", "
FMT += '"m":"%(message)s"'
FMT += "}"


def LoggerStream(stream=sys.stdout):
    lggr = logging.Logger(name="single-use-for-print")
    fmtr = logging.Formatter(fmt=FMT, datefmt=DATEFMT_ISO8601)
    stha = logging.StreamHandler(stream)
    stha.setFormatter(fmtr)
    lggr.addHandler(stha)
    lggr.setLevel(logging.DEBUG)
    return lggr


def LoggerFile(path):
    lggr = logging.Logger(name=path)
    file_handler = logging.FileHandler(filename=path, mode="w")
    fmtr = logging.Formatter(fmt=FMT, datefmt=DATEFMT_ISO8601)
    file_handler.setFormatter(fmtr)
    lggr.addHandler(file_handler)
    lggr.setLevel(logging.DEBUG)
    return lggr


class TimeDelta:
    def __init__(self, logger, name, level=logging.INFO):
        self.logger = logger
        self.name = name

    def __enter__(self):
        self.start = time.time()
        self.logger.log(
            level=level, msg="{:s}:start:{:f}".format(self.name, self.start)
        )
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop = time.time()
        self.logger.log(
            level=level, msg="{:s}:stop:{:f}".format(self.name, self.start)
        )
        self.logger.log(
            level=level, msg="{:s}:delta:{:f}".format(self.name, self.delta())
        )

    def delta(self):
        return self.stop - self.start


def reduce(list_of_log_paths, out_path):
    log_records = reduce_into_records(list_of_log_paths=list_of_log_paths)
    log_df = pd.DataFrame(log_records)
    log_df = log_df.sort_values(by=["run_id"])
    log_df.to_csv(out_path + ".tmp", index=False, na_rep="nan")
    shutil.move(out_path + ".tmp", out_path)


def reduce_into_records(list_of_log_paths):
    list_of_log_records = []
    for log_path in list_of_log_paths:
        run_id = int(os.path.basename(log_path)[0:6])
        run = {"run_id": run_id}

        key = ":delta:"
        with open(log_path, "rt") as fin:
            for line in fin:
                logline = json_numpy.loads(line)
                if "msg" in logline:
                    msg = logline["msg"]
                    if key in msg:
                        iname = str.find(msg, key)
                        name = msg[:(iname)]
                        deltastr = msg[(iname + len(key)) :]
                        run[name] = float(deltastr)
            list_of_log_records.append(run)

    return list_of_log_records


class MapAndReducePoolWithLogger:
    def __init__(pool, logger):
        self.pool = pool
        self.logger = logger

    def accepts_logger(self):
        signature = inspect.signature(self.pool.map)
        return "logger" in signature.parameters

    def map(self, function, jobs):
        if self.accepts_logger():
            return self.pool.map(function, jobs, logger=self.logger)
        else:
            return self.pool.map(function, jobs)
