import logging
from contextlib import contextmanager


@contextmanager
def scheduler_context(args):
    """Set the scheduler to use, based on the script arguments"""

    import dask

    sched_info = {}

    try:
        if args.scheduler in ("mt", "thread", "threads", "threaded", "threading"):
            logging.info("Using multithreaded scheduler")
            dask.config.set(scheduler="threads")
            sched_info = {"type": "threaded"}
        elif args.scheduler in ("mp", "processes", "multiprocessing"):
            raise ValueError(
                "The Process Scheduler does not currently " "work with dask-ms"
            )
            import dask.multiprocessing

            logging.info("Using multiprocessing scheduler")
            dask.config.set(scheduler="processes")
            sched_info = {"type": "multiprocessing"}
        else:
            import distributed

            local_cluster = None

            if args.scheduler == "local":
                local_cluster = distributed.LocalCluster(processes=False)
                address = local_cluster.scheduler_address
            elif args.scheduler.startswith("tcp"):
                address = args.scheduler
            else:
                import json

                with open(args.scheduler, "r") as f:
                    address = json.load(f)["address"]

            logging.info(f"Using distributed scheduler with address '{address}'")
            client = distributed.Client(address)
            dask.config.set(scheduler=client)
            client.restart()
            sched_info = {
                "type": "distributed",
                "client": client,
                "local_cluster": local_cluster,
            }

        yield
    except Exception:
        logging.exception("Error setting up scheduler", exc_info=True)

    finally:
        try:
            sched_type = sched_info["type"]
        except KeyError:
            pass
        else:
            if sched_type == "distributed":
                try:
                    client = sched_info["client"]
                except KeyError:
                    pass
                else:
                    client.close()

                try:
                    local_cluster = sched_info["local_cluster"]
                except KeyError:
                    pass
                else:
                    local_cluster.close()
