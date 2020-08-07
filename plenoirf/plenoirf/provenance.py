import os
import subprocess
import datetime
import warnings
import shutil
import corsika_primary_wrapper


IMPORTANT_PROGRAMS = {
    "git": {"version": "--version"},
    "python": {"version": "--version"},
    "cmake": {"version": "--version"},
    "make": {"version": "--version"},
    "gcc": {"version": "--version"},
    "g++": {"version": "--version"},
    "f77": {"version": "--version"},
}


def _get_ascii_stdout_stderr(command, cwd='.'):
    pp = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
    )
    so, se = pp.communicate()
    return so.decode('ascii'), se.decode('ascii')


def get_ascii_stdout_stderr(command, cwd='.'):
    try:
        return _get_ascii_stdout_stderr(command=command, cwd=cwd)
    except Exception as e:
        print(e)
        warnings.warn("Failed to get stdout and stderr.")
        return ("", "")


def _git_last_commit_hash(path):
    o, _ = get_ascii_stdout_stderr(
        command=['git', 'log', '-1'],
        cwd=path,
    )
    lines = o.splitlines()
    firstline = lines[0]
    commit_hash = firstline.split(' ')[1]
    assert len(commit_hash) > 0
    return commit_hash


def git_last_commit_hash(path):
    try:
        return _git_last_commit_hash(path)
    except Exception as e:
        print(e)
        warnings.warn("Failed to get git's last commit-hash.")
        return ""


def get_time_dict_now():
    dt = datetime.datetime.now()
    return {
        "unix": float(dt.timestamp()),
        "iso": dt.isoformat(),
    }


def get_hostname():
    try:
        import socket
        return socket.gethostname()
    except Exception as e:
        print(e)
        warnings.warn("Failed to get hostname.")
        return ""


def get_username():
    try:
        import getpass
        return getpass.getuser()
    except Exception as e:
        print(e)
        warnings.warn("Failed to get username.")
        return ""


def which(programname):
    try:
        return os.path.abspath(shutil.which(programname))
    except Exception as e:
        print(e)
        warnings.warn("Failed to find program {:s}.".format(str(programname)))
        return ""


def starter_kit_abspath():
    # Expect the corsika_primary_wrapper to be in the "starter_kit"
    _p = os.path.abspath(corsika_primary_wrapper.__file__)
    for i in range(4):
        _p = os.path.split(_p)[0]
    return _p



def get_current_working_directory():
    return os.getcwd()


def make_provenance():
    p = {}
    p['time'] = get_time_dict_now()
    p['hostname'] = get_hostname()
    p['username'] = get_username()
    p['current_working_directory'] = get_current_working_directory()
    p['which'] = {}
    for prg in IMPORTANT_PROGRAMS:
        p['which'][prg] = which(prg)

    p['version'] = {}
    for prg in IMPORTANT_PROGRAMS:
        _o, _ = get_ascii_stdout_stderr(
            command=[prg, IMPORTANT_PROGRAMS[prg]['version']]
        )
        p['version'][prg] = _o.splitlines()[0]

    p["starter_kit"] = {}
    p["starter_kit"]["path"] = starter_kit_abspath()

    p["starter_kit"]["git"] = {}
    p["starter_kit"]["git"]["commit"] = git_last_commit_hash(
        path=p["starter_kit"]["path"]
    )
    p["starter_kit"]["git"]["status"] = get_ascii_stdout_stderr(
        command=['git', 'status'],
        cwd=p["starter_kit"]["path"]
    )[0]
    p["starter_kit"]["git"]["diff"] = get_ascii_stdout_stderr(
        command=['git', 'diff', '--submodule=diff'],
        cwd=p["starter_kit"]["path"]
    )[0]




    return p
