# Codeforces Gym 106478C: Run, Fix, Repeat

This is an interactive optimization problem with partial scoring.

## Setting

A hidden test is a `125 x 125` grid. Each cell is either:

- `#`: a server rack, which the robot cannot use;
- `.`: an empty cell, where the repair point may be located.

The hidden grid is produced by the official generator from `problem-c-generator.zip`.
Each generated test starts with:

```text
N T
```

For this problem, `N = 125` and `T = 10000`. The next `N` lines contain the hidden
grid. During the real interaction the contestant program does not receive these
grid lines; they are used only by the interactor.

## Interaction Protocol

At the start, the contestant program reads:

```text
N T
```

Then it may make at most `T` queries. A query is two 1-based coordinates:

```text
r c
```

The interactor replies with one integer:

- `-1` if `(r, c)` is a server rack;
- `0` if `(r, c)` currently contains the repair point;
- otherwise, the shortest path distance from `(r, c)` to the current repair
  point, moving only through empty cells.

Whenever the reply is `0`, the contestant receives one point for that test, and
the repair point is immediately relocated to a uniformly random empty cell. It
may be relocated to the same cell again.

The contestant may stop early by printing:

```text
-1 -1
```

## Goal

Maximize the number of times the repair point is found within the `10000` query
limit.

The official test set has `50` generated grids. The repair point randomness is
fresh on each run, so identical submissions can receive different raw scores.

## Scoring

For one test, the raw score is the number of successful finds. Invalid
interaction, runtime errors, timeouts, or malformed queries give `0` for that
run.

Across the official tests:

```text
total_points = sum(raw_points over all tests)
final_score = 800 * total_points / best_total_points
```

In this local workspace, the validator estimates the score from average local
performance. It uses `90053` as `best_total_points` and scales the average run
to the official `50` tests:

```text
local_score = 800 * average_points_per_run * 50 / 90053
```

By default, `validator.py` generates 10 fresh maps, runs the solution once on
each map, and reports this average score. It is useful for comparing local
experiments, not as an exact Codeforces score prediction.

## Provided Local Materials

- `materials/problem-c-generator.zip`: official Codeforces generator archive.
- `materials/problem-c-generator/files/generator.cpp`: extracted generator.
- `validator.py`: local interactive runner for Python solutions. Without
  `--maps`, it creates fresh maps with random generator seeds and prints those
  seeds so individual runs can be replayed with `--map-seed`.
