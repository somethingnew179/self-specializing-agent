#!/usr/bin/env python3
"""Validate generated Lost Cursor outputs."""

from __future__ import annotations

import argparse
import os

from solve import CursorSim, read_png


ABSOLUTE_OPTIMUMS = {
    "01": 390,
    "02": 208,
    "03": 1210,
    "04": 392,
    "05": 101,
    "06": 57,
    "07": 24,
    "08": 233,
}


def case_score(case: str, ok: bool, length: int) -> float:
    if not ok or length <= 0:
        return 0.0
    return 100.0 * (ABSOLUTE_OPTIMUMS[case] / length) ** 2


def validate_output(case: str, outputs_dir: str, inputs_dir: str) -> tuple[bool, int, int]:
    path = os.path.join(outputs_dir, f"{case}.out")
    with open(path, "r", encoding="ascii") as f:
        sequence = f.readline().strip()

    if not sequence or len(sequence) > 5000 or any(ch not in "UDLR" for ch in sequence):
        return False, len(sequence), -1

    width, height, rows = read_png(os.path.join(inputs_dir, f"{case}.png"))
    remaining = CursorSim(width, height, rows).remaining(sequence)
    return remaining == 0, len(sequence), remaining


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default="tests/inputs")
    parser.add_argument("--outputs", default="outputs")
    args = parser.parse_args()

    all_ok = True
    total = 0
    total_score = 0.0
    for i in range(1, 9):
        case = f"{i:02d}"
        ok, length, remaining = validate_output(case, args.outputs, args.inputs)
        total += length if ok else 0
        score = case_score(case, ok, length)
        total_score += score
        all_ok &= ok
        status = "ok" if ok else f"invalid remaining={remaining}"
        print(f"{case}: {length} {status} score={score:.6f}")
    print(f"total_valid_length: {total}")
    print(f"total_score: {total_score:.6f}")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
