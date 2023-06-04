# !/usr/bin/env python3

import os
import sys
import datetime

original_print = print


def _stdout_write(s, flush=False):
    sys.stdout.write(s)
    if flush:
        sys.stdout.flush()


def _stderr_write(s, flush=False):
    sys.stderr.write(s)
    if flush:
        sys.stderr.flush()


def _debug_print(*args, sep=' ', end='\n', file=None, flush=True):
    args = (str(arg) for arg in args)  # convert to string as numbers cannot be joined
    if file == sys.stderr:
        _stderr_write(sep.join(args), flush)
    elif file in [sys.stdout, None]:
        lineno = sys._getframe().f_back.f_lineno
        filename = sys._getframe(1).f_code.co_filename

        stdout = f'\033[31m{datetime.datetime.now().strftime("%H:%M:%S.%f")}\x1b[0m  \033[32m{filename}:{lineno}\x1b[0m  {sep.join(args)} {end}'
        _stdout_write(stdout, flush)
    else:
        # catch exceptions
        original_print(*args, sep=sep, end=end, file=file)


# monkey patch print
def patch_print():
    try:
        __builtins__.print = _debug_print
    except AttributeError:
        __builtins__['print'] = _debug_print


def remove_patch_print():
    try:
        __builtins__.print = original_print
    except AttributeError:
        __builtins__['print'] = original_print