import os
from subprocess import call
import pytest

def test_all_mctracer_Cpp_test_cases(capsys):
    cwd = os.getcwd()
    os.chdir(os.path.join('mctracer', 'Tests'))
    with capsys.disabled():
        rtc = call([os.path.join('..', '..','build', 'mctracer', 'mctTest')])
    os.chdir(cwd)
    assert rtc == 0