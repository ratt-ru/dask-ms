from os.path import join as pjoin

collect_ignore = ["setup.py"]
collect_ignore += [pjoin('xarrayms', 'tests', 'test_correct_read.py'),
                   pjoin('xarrayms', 'tests', 'test_correct_write.py'),
                   pjoin('xarrayms', 'tests', 'test_zarr_write.py')]
