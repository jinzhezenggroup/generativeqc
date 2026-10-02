"""PBE0/def2-SVP energy and analytic forces, using the README HF protocol.

Both engines use the same explicit moving grid and offline spherical basis.
No VV10 adapter, density fitting, mixed precision or opt-in schedule is used.
"""

from benchmarks.readme_omol25 import EndpointSpec
from benchmarks.readme_omol25 import main as run_endpoint

SCHEMA = "generativeqc.readme-pbe0.v1"
PBE0 = EndpointSpec("pbe0", "def2-SVP", SCHEMA)


def main() -> None:
    """Retain independent cold/moved oracles and all five fixed-density replays."""
    run_endpoint(PBE0)


if __name__ == "__main__":
    main()
