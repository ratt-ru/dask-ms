# -*- coding: utf-8 -*-
collect_ignore = ["setup.py"]


def pytest_addoption(parser):
    parser.addoption(
        "--stress",
        action="store_true",
        dest="stress",
        default=False,
        help="Enable stress tests",
    )
    parser.addoption(
        "--optional",
        action="store_true",
        dest="optional",
        default=False,
        help="Enable optional tests",
    )
    parser.addoption(
        "--applications",
        action="store_true",
        dest="applications",
        default=False,
        help="Enable application tests",
    )


def pytest_configure(config):
    # Add non-standard markers
    config.addinivalue_line("markers", "stress: long running stress tests")
    config.addinivalue_line("markers", "optional: optional tests")
    config.addinivalue_line("markers", "applications: application tests")

    markexpr = [config.option.markexpr] if config.option.markexpr else []

    for mark in ("stress", "optional", "applications"):
        test = "" if getattr(config.option, mark, False) else "not "
        markexpr.append(f"{test}{mark}")

    config.option.markexpr = " and ".join(markexpr)
    print(config.option.markexpr)
