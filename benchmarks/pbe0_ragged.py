"""Default Ragged XC versus explicit opt-out, complete cold/warm/moved E+F.

Both arms retain the ordinary SCF controller, exact J/K and scientific protocol.
Diagnostic trajectories attest batch admission separately from clean timings.
"""

from benchmarks.pbe0_xc_trajectories import main

if __name__ == "__main__":
    main(strategy="ragged")
