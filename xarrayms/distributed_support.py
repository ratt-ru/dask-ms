"""
Somewhat hacky technique of providing thread
secession and rejoin functionality when a thread
is scheduled on a dask distributed worker
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import distributed
except ImportError:
    # Not installed, secede and rejoin are noops
    def secede(adjust=True):
        pass

    def rejoin():
        pass
else:
    from distributed.threadpoolexecutor import (thread_state,
                                                secede as dist_secede,
                                                rejoin as dist_rejoin)

    # If thread_state has a proceed member, then it's executing
    # in dask distributed threadpool executor
    def secede(adjust=True):
        if hasattr(thread_state, "proceed"):
            dist_secede(adjust=adjust)

    def rejoin():
        if hasattr(thread_state, "proceed"):
            dist_rejoin()
