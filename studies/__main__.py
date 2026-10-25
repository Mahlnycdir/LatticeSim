"""
Run all studies, save their results to results/ and print the report.

    python -m studies                 # all studies
    python -m studies sectors spectra # selected studies
"""
import sys
import time

from studies import correlators, free_field, potentials, sectors, spectra

STUDIES = {"free_field": free_field, "spectra": spectra, "correlators": correlators,
           "sectors": sectors, "potentials": potentials}


def main(names):
    unknown = set(names) - set(STUDIES)
    if unknown:
        sys.exit(f"Unknown studies {sorted(unknown)}; choose from {list(STUDIES)}.")
    for name in names or STUDIES:
        start = time.perf_counter()
        module = STUDIES[name]
        module.report(module.run())
        print(f"  [{name}: {time.perf_counter() - start:.1f} s, results in results/{name}/]")


if __name__ == "__main__":
    main(sys.argv[1:])
