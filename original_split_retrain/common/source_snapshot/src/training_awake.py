"""Prevent automatic system sleep only for the lifetime of a training task.

Does not keep the display on, change power-plan settings, or override manual sleep.
"""
import argparse
import ctypes
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _kernel():
    if os.name != 'nt':
        raise RuntimeError('This power guard is for Windows')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
    kernel.SetThreadExecutionState.restype = ctypes.c_uint32
    return kernel


def _status(path, **fields):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'guard_pid': os.getpid(), 'updated_at': datetime.now().astimezone().isoformat(),
               'display_required': False, 'manual_sleep_overridden': False, **fields}
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


@contextmanager
def keep_system_awake(status_path=None):
    kernel = _kernel()
    previous = kernel.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    if not previous:
        raise ctypes.WinError(ctypes.get_last_error())
    observed = kernel.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    if not observed & ES_SYSTEM_REQUIRED:
        kernel.SetThreadExecutionState(ES_CONTINUOUS)
        raise RuntimeError('Windows did not retain the system-awake request')
    _status(status_path, state='active', windows_request_verified=True,
            flags=hex(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
    try:
        yield
    finally:
        cleared = kernel.SetThreadExecutionState(ES_CONTINUOUS)
        _status(status_path, state='released', windows_request_verified=bool(cleared))


def watch_process(pid, status_path):
    kernel = _kernel()
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel.WaitForSingleObject.restype = ctypes.c_uint32
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int
    # A real process handle remains bound to this process, even if its PID is reused.
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        with keep_system_awake(status_path):
            while True:
                state = kernel.WaitForSingleObject(handle, 30000)
                if state == 0:
                    break
                if state != 258:
                    raise ctypes.WinError(ctypes.get_last_error())
                _status(status_path, state='active', watched_pid=pid,
                        windows_request_verified=True, flags=hex(ES_CONTINUOUS | ES_SYSTEM_REQUIRED))
    finally:
        kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch-pid', type=int)
    parser.add_argument('--status-file', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        kernel = _kernel()
        with keep_system_awake(args.status_file):
            state = kernel.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            assert state & ES_SYSTEM_REQUIRED
        state = kernel.SetThreadExecutionState(ES_CONTINUOUS)
        assert not state & ES_SYSTEM_REQUIRED
        print('POWER_GUARD_VERIFIED: request set and released; display not held on')
    elif args.watch_pid:
        watch_process(args.watch_pid, args.status_file)
    else:
        parser.error('--watch-pid or --self-test is required')


if __name__ == '__main__':
    main()
