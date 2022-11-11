from appdirs import user_cache_dir
from hashlib import sha256
import logging
from pathlib import Path
import tarfile

import pytest

log = logging.getLogger(__file__)

TAU_MS = "HLTau_B6cont.calavg.tav300s"
TAU_MS_TAR = f"{TAU_MS}.tar.xz"
TAU_MS_TAR_HASH = "fc2ce9261817dfd88bbdd244c8e9e58ae0362173938df6ef2a587b1823147f70"
DATA_URL = f"s3://ratt-public-data/test-data/{TAU_MS_TAR}"


def download_tau_ms(tau_ms_tar):
    if tau_ms_tar.exists():
        with open(tau_ms_tar, "rb") as f:
            digest = sha256()

            while data := f.read(2**20):
                digest.update(data)

            if digest.hexdigest() == TAU_MS_TAR_HASH:
                return

            tau_ms_tar.unlink(missing_ok=True)
            raise ValueError(
                f"sha256 digest '{digest.hexdigest()}' "
                f"of {tau_ms_tar} does not match "
                f"{TAU_MS_TAR_HASH}"
            )
    else:
        s3fs = pytest.importorskip("s3fs")
        s3 = s3fs.S3FileSystem(anon=True)

        for attempt in range(3):
            with s3.open(DATA_URL, "rb") as fin, open(tau_ms_tar, "wb") as fout:
                digest = sha256()

                while data := fin.read(2**20):
                    digest.update(data)
                    fout.write(data)

                if digest.hexdigest() == TAU_MS_TAR_HASH:
                    return

                log.warning("Download of %s failed on attempt %d", DATA_URL, attempt)
                tau_ms_tar.unlink(missing_ok=True)

        raise ValueError(f"Download of {DATA_URL} failed {attempt} times")


@pytest.fixture(scope="function")
def tau_ms(tmp_path_factory):
    cache_dir = Path(user_cache_dir("dask-ms")) / "test-data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    tau_ms_tar = cache_dir / TAU_MS_TAR

    download_tau_ms(tau_ms_tar)
    msdir = tmp_path_factory.mktemp("taums")

    with tarfile.open(tau_ms_tar) as tar:
        tar.extractall(msdir)

    yield msdir / TAU_MS
