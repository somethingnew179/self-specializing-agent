# Local Tools for Codeforces Gym 106478C

This workspace contains a local statement rewrite, the official generator archive
downloaded from the Codeforces mirror, and a Python validator.

## Files

- `statement.md`: English problem statement rewrite.
- `materials/problem-c-generator.zip`: official generator archive.
- `materials/problem-c-generator/files/generator.cpp`: extracted generator source.
- `validator.py`: local interactor/scorer for Python solutions. By default it
  runs 10 fresh generated maps and reports the average score.

## Build the Official Generator

```bash
make generator
```

## Validate a Python Solution

```bash
python3 validator.py path/to/solution.py
```

By default, this builds/uses `build/problem-c-generator`, creates 10 fresh maps
with new random generator seeds, runs the solution once on each map, and prints
both:

- `map_seed`: the generator seed, replayable with `--map-seed`;
- `repair_seed`: the local random seed for repair-point relocations, replayable
  with `--rng-seed`.

The reported score is the average of those 10 validator runs, scaled to the
official 50-test format. It uses `90053` as the best total:

```text
score = 800 * average_points * 50 / 90053
```

Useful options:

```bash
python3 validator.py solution.py --generated-tests 20
python3 validator.py solution.py --generated-tests 5 --runs-per-map 3
python3 validator.py solution.py --map-seed b6a95da001f292fa30e1be123b757518
python3 validator.py solution.py --map-seed b6a95da001f292fa30e1be123b757518 --rng-seed 788592913490958267
python3 validator.py solution.py --maps saved/*.in
python3 validator.py solution.py --show-stderr
```

`--maps` switches back to fixed map files and disables fresh generation. Use it
only when you want to debug specific saved `.in` files.

The local score is an experiment score over the maps you ran locally. With the
default fresh generation mode it is not the exact official Codeforces score,
because the official contest uses its own fixed test set.
