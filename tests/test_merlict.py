import os
from subprocess import call
import pytest


def test_merlict(capsys):
    cwd = os.getcwd()
    os.chdir(os.path.join('merlict_development_kit'))
    with capsys.disabled():
        rtc = call([os.path.join('..', 'build', 'merlict', 'merlict-test')])
    os.chdir(cwd)
    assert rtc == 0
