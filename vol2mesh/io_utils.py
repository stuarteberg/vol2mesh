import os
import sys
from contextlib import contextmanager

import ctypes
from ctypes.util import find_library
try:
    libc = ctypes.cdll.msvcrt # Windows
except (OSError, AttributeError):
    libc = ctypes.cdll.LoadLibrary(find_library('c'))


@contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    Context manager.

    Redirects a file object or file descriptor to a new file descriptor.

    Example:
    with open('my-stdout.txt', 'w') as f:
        with stdout_redirected(f):
            print('Writing to my-stdout.txt')

    Motivation: In pure-Python, you can redirect all print() statements like this:

        sys.stdout = open('myfile.txt')

        ...but that doesn't redirect any compiled printf() (or std::cout) output
        from C/C++ extension modules.

    This context manager uses a superior approach, based on low-level Unix file
    descriptors, which redirects both Python AND C/C++ output.

    Lifted from the following link (with minor edits):
    https://stackoverflow.com/a/22434262/162094
    (MIT License)
    """
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)

    try:
        if fileno(to) == stdout_fd:
            # Nothing to do; early return
            yield stdout
            return
    except ValueError:  # filename
        pass

    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        flush(stdout)  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            flush(stdout)
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


def flush(stream):
    try:
        libc.fflush(None)  # Flush all C stdio buffers
        stream.flush()
    except (AttributeError, ValueError, IOError):
        pass  # unsupported


def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd
