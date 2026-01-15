from daskms.constants import DASKMS_PARTITION_KEY

TAG_TO_INTENT = {
    "gaincal": "CALIBRATE_PHASE,CALIBRATE_AMPLI",
    "bpcal": "CALIBRATE_BANDPASS,CALIBRATE_FLUX",
    "target": "TARGET",
}


# Partitioning columns
GROUP_COLS = ["FIELD_ID", "DATA_DESC_ID", "SCAN_NUMBER"]

# No partitioning, applies to many subtables
EMPTY_PARTITION_SCHEMA = {DASKMS_PARTITION_KEY: ()}

# katdal datasets only have one spectral window
# and one polarisation. Thus, there
# is only one DATA_DESC_ID and it is zero
DATA_DESC_ID = 0
