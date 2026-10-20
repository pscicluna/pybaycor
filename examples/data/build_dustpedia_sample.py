"""Rebuild ``dustpedia_sample.csv`` from the DustPedia catalogues at CDS.

The example notebook reads the CSV written by this script, so it runs offline. Run it
again (``python examples/data/build_dustpedia_sample.py``) to regenerate the file from
the original tables:

* Nersesian et al. (2019, A&A 624, A80; CDS J/A+A/624/A80): CIGALE fits with the
  THEMIS dust model, giving stellar masses, star-formation rates, dust masses and dust
  temperatures with their uncertainties.
* De Vis et al. (2019, A&A 623, A5; CDS J/A+A/623/A5): global HI masses, with a flag
  that marks upper limits (non-detections) and confused sources.

Only the columns used by the notebook are kept. Quantities are copied unchanged from
the catalogues (linear units); the notebook takes logarithms and propagates the
uncertainties itself.
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

CDS = "https://cdsarc.cds.unistra.fr/ftp"
OUT = Path(__file__).with_name("dustpedia_sample.csv")

# (column name, first byte, last byte), from the catalogues' ReadMe files.
THEMIS = [
    ("name", 1, 23),
    ("hubble_type", 45, 48),
    ("dist_mpc", 50, 61),
    ("sfr", 63, 70),
    ("e_sfr", 72, 79),
    ("mstar", 81, 89),
    ("e_mstar", 91, 99),
    ("mdust", 205, 213),
    ("e_mdust", 215, 223),
    ("tdust", 259, 264),
    ("e_tdust", 266, 271),
]
HI = [
    ("name", 1, 23),
    ("mhi", 126, 149),
    ("e_mhi", 151, 174),
    ("hi_flag", 238, 238),
]


def read_cds(path: str, spec: list[tuple[str, int, int]]) -> pd.DataFrame:
    """Read a fixed-width CDS table, keeping the columns in ``spec``."""
    with urllib.request.urlopen(f"{CDS}/{path}", timeout=60) as response:
        text = response.read().decode()
    return pd.read_fwf(
        io.StringIO(text),
        colspecs=[(first - 1, last) for _, first, last in spec],
        names=[name for name, _, _ in spec],
        na_values=["-", "---"],
    )


def main() -> None:
    """Download the catalogues, merge them and write the CSV."""
    sed = read_cds("J/A+A/624/A80/themis.dat", THEMIS)
    hi = read_cds("J/A+A/623/A5/dp-hi.dat", HI)
    sample = sed.merge(hi, on="name", how="left", validate="one_to_one")
    # Galaxies without a CIGALE fit have no stellar mass, SFR or dust mass.
    sample = sample[sample["mstar"].notna()].reset_index(drop=True)
    sample["hi_flag"] = sample["hi_flag"].astype("Int64")
    with OUT.open("w") as f:
        f.write(
            "# DustPedia galaxies with CIGALE/THEMIS fits (Nersesian+2019, J/A+A/624/A80)\n"
            "# and global HI masses (De Vis+2019, J/A+A/623/A5), built by\n"
            "# build_dustpedia_sample.py. Units: Msun, Msun/yr, K, Mpc.\n"
            "# hi_flag: 0 detected, 1 upper limit (mhi is the limit), 2 confused,\n"
            "# 3 detected and corrected for flux outside the ALFALFA beam; empty: no HI data.\n"
        )
        sample.to_csv(f, index=False, float_format="%.6g")
    print(f"wrote {len(sample)} galaxies to {OUT}")


if __name__ == "__main__":
    main()
