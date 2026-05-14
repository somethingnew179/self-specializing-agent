#!/usr/bin/env python3
"""Local interactor and scorer for Codeforces Gym 106478C.

The validator runs a Python solution as an interactive subprocess, feeds it the
public `N T` header, emulates the hidden grid responses, and reports raw points
plus the average normalized score over 10 fresh runs by default.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import glob
import os
import random
import secrets
import select
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_GENERATOR = Path("build/problem-c-generator")
DEFAULT_BEST_TOTAL = 90053
DEFAULT_OFFICIAL_TESTS = 50
DEFAULT_VALIDATOR_RUNS = 10
DEFAULT_TIMEOUT = 2.0
DEFAULT_GENERATOR_TIMEOUT = 5.0


class ProtocolError(Exception):
    pass


@dataclass(frozen=True)
class MapCase:
    name: str
    n: int
    t: int
    grid: tuple[str, ...]
    empty: tuple[tuple[int, int], ...]
    map_seed: str | None = None


@dataclass
class RunResult:
    case_name: str
    trial: int
    map_seed: str | None
    repair_seed: int
    points: int
    queries: int
    status: str
    message: str = ""
    elapsed: float = 0.0


def parse_map(path: Path) -> MapCase:
    return parse_map_lines(path.name, path.read_text(encoding="utf-8").splitlines())


def parse_map_lines(source: str, lines: list[str], map_seed: str | None = None) -> MapCase:
    if not lines:
        raise ValueError(f"{source}: empty map file")

    first = lines[0].split()
    if len(first) != 2:
        raise ValueError(f"{source}: first line must contain N and T")

    n, t = map(int, first)
    if len(lines) < n + 1:
        raise ValueError(f"{source}: expected {n} grid lines, found {len(lines) - 1}")

    grid = tuple(lines[1 : n + 1])
    for i, row in enumerate(grid, start=1):
        if len(row) != n:
            raise ValueError(f"{source}: row {i} has length {len(row)}, expected {n}")
        bad = sorted(set(row) - {".", "#"})
        if bad:
            raise ValueError(f"{source}: row {i} contains unsupported characters {bad}")

    empty = tuple((r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == ".")
    if not empty:
        raise ValueError(f"{source}: map has no empty cells")

    return MapCase(source, n, t, grid, empty, map_seed)


def fresh_map_seed() -> str:
    return secrets.token_hex(16)


def fresh_repair_seed() -> int:
    return secrets.randbits(64)


def ensure_generator(generator: Path) -> None:
    if generator.is_file():
        return

    makefile = Path("Makefile")
    if makefile.is_file():
        subprocess.run(["make", "generator"], check=True)

    if not generator.is_file():
        raise FileNotFoundError(
            f"generator binary not found: {generator}. Build it with `make generator` "
            "or pass --generator PATH."
        )


def generate_map(generator: Path, seed: str, timeout: float) -> MapCase:
    completed = subprocess.run(
        [str(generator), seed],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"generator failed for seed {seed!r} with code {completed.returncode}: {stderr}")

    return parse_map_lines(f"seed-{seed}.in", completed.stdout.splitlines(), map_seed=seed)


def bfs_distances(case: MapCase, target: tuple[int, int]) -> list[list[int]]:
    n = case.n
    dist = [[-1] * n for _ in range(n)]
    tr, tc = target
    dist[tr][tc] = 0
    q: deque[tuple[int, int]] = deque([(tr, tc)])

    while q:
        r, c = q.popleft()
        nd = dist[r][c] + 1
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < n and 0 <= nc < n and case.grid[nr][nc] == "." and dist[nr][nc] < 0:
                dist[nr][nc] = nd
                q.append((nr, nc))

    return dist


def set_nonblocking(file_obj: object) -> None:
    fd = file_obj.fileno()
    flags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)


def read_line(proc: subprocess.Popen[bytes], buffer: bytearray, timeout: float) -> str | None:
    deadline = time.monotonic() + timeout
    fd = proc.stdout.fileno()

    while True:
        newline = buffer.find(b"\n")
        if newline >= 0:
            raw = bytes(buffer[:newline])
            del buffer[: newline + 1]
            return raw.decode("utf-8", errors="replace").strip()

        if proc.poll() is not None:
            if buffer:
                raw = bytes(buffer)
                buffer.clear()
                return raw.decode("utf-8", errors="replace").strip()
            return None

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProtocolError(f"timeout while waiting for query after {timeout:.3f}s")

        readable, _, _ = select.select([fd], [], [], remaining)
        if not readable:
            continue

        try:
            chunk = os.read(fd, 4096)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                continue
            raise

        if not chunk:
            if buffer:
                raw = bytes(buffer)
                buffer.clear()
                return raw.decode("utf-8", errors="replace").strip()
            return None

        buffer.extend(chunk)


def parse_query(line: str, query_index: int) -> tuple[int, int]:
    parts = line.split()
    if len(parts) != 2:
        raise ProtocolError(f"query {query_index}: expected two integers, got {line!r}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ProtocolError(f"query {query_index}: expected integers, got {line!r}") from exc


def write_response(proc: subprocess.Popen[bytes], value: int) -> None:
    try:
        proc.stdin.write(f"{value}\n".encode("ascii"))
        proc.stdin.flush()
    except BrokenPipeError as exc:
        raise ProtocolError("solution terminated while the interactor was sending a response") from exc


def drain_stdout(proc: subprocess.Popen[bytes], buffer: bytearray) -> str:
    fd = proc.stdout.fileno()
    chunks = [bytes(buffer)]
    buffer.clear()

    while True:
        readable, _, _ = select.select([fd], [], [], 0)
        if not readable:
            break
        try:
            chunk = os.read(fd, 4096)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                break
            raise
        if not chunk:
            break
        chunks.append(chunk)

    return b"".join(chunks).decode("utf-8", errors="replace")


def terminate_process(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=0.5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=0.5)


def run_one(
    solution: Path,
    case: MapCase,
    trial: int,
    rng_seed: int,
    python: str,
    timeout: float,
    allow_eof_stop: bool,
    show_stderr: bool,
) -> RunResult:
    start = time.monotonic()
    rng = random.Random(rng_seed)
    target = rng.choice(case.empty)
    dist = bfs_distances(case, target)

    proc = subprocess.Popen(
        [python, str(solution)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    assert proc.stderr is not None
    set_nonblocking(proc.stdout)

    points = 0
    queries = 0
    status = "OK"
    message = ""
    out_buffer = bytearray()

    try:
        proc.stdin.write(f"{case.n} {case.t}\n".encode("ascii"))
        proc.stdin.flush()

        for query_index in range(1, case.t + 1):
            line = read_line(proc, out_buffer, timeout)
            if line is None:
                if allow_eof_stop:
                    status = "OK"
                    message = "solution stopped by EOF"
                    break
                raise ProtocolError("solution ended without printing -1 -1")

            if not line:
                raise ProtocolError(f"query {query_index}: empty line")

            r1, c1 = parse_query(line, query_index)
            if (r1, c1) == (-1, -1):
                status = "OK"
                message = "solution stopped with -1 -1"
                break

            queries += 1
            if not (1 <= r1 <= case.n and 1 <= c1 <= case.n):
                raise ProtocolError(f"query {query_index}: coordinates out of range: {r1} {c1}")

            r = r1 - 1
            c = c1 - 1
            if case.grid[r][c] == "#":
                write_response(proc, -1)
                continue

            if (r, c) == target:
                points += 1
                write_response(proc, 0)
                target = rng.choice(case.empty)
                dist = bfs_distances(case, target)
                continue

            value = dist[r][c]
            if value < 0:
                raise ProtocolError(f"query {query_index}: empty cell is disconnected from repair point")
            write_response(proc, value)

        else:
            status = "OK"
            message = "query limit reached"

        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise ProtocolError("solution did not terminate after interaction ended")

        if proc.returncode not in (0, None):
            raise ProtocolError(f"solution exited with code {proc.returncode}")

        extra_output = drain_stdout(proc, out_buffer)
        if extra_output.strip():
            raise ProtocolError(f"solution printed extra output after interaction ended: {extra_output!r}")

    except ProtocolError as exc:
        points = 0
        status = "INVALID"
        message = str(exc)
        terminate_process(proc)
    finally:
        elapsed = time.monotonic() - start

    stderr_text = ""
    try:
        stderr_text = proc.stderr.read().decode("utf-8", errors="replace")
    except Exception:
        stderr_text = ""
    if show_stderr and stderr_text.strip():
        sys.stderr.write(f"\n--- stderr from {case.name} trial {trial} ---\n{stderr_text}\n")

    return RunResult(case.name, trial, case.map_seed, rng_seed, points, queries, status, message, elapsed)


def expand_maps(patterns: Iterable[str]) -> list[Path]:
    result: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            result.extend(Path(path) for path in matches)
        else:
            result.append(Path(pattern))
    return sorted(dict.fromkeys(result), key=map_sort_key)


def map_sort_key(path: Path) -> tuple[str, int, str]:
    try:
        return (str(path.parent), int(path.stem), path.suffix)
    except ValueError:
        return (str(path.parent), sys.maxsize, path.name)


def print_result(result: RunResult) -> None:
    suffix = f" - {result.message}" if result.message else ""
    seed = f" map_seed={result.map_seed}" if result.map_seed is not None else ""
    print(
        f"{result.case_name} trial {result.trial:>2}{seed} repair_seed={result.repair_seed}: "
        f"{result.status:<7} points={result.points:<5} queries={result.queries:<5} "
        f"time={result.elapsed:.3f}s{suffix}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Python solution against fresh local 106478C maps.")
    parser.add_argument("solution", type=Path, help="path to the Python solution to validate")
    parser.add_argument(
        "--maps",
        nargs="+",
        default=None,
        help="optional fixed map files or glob patterns; if omitted, fresh maps are generated",
    )
    parser.add_argument(
        "--generated-tests",
        type=int,
        default=DEFAULT_VALIDATOR_RUNS,
        help="number of fresh validator runs when --maps is omitted",
    )
    parser.add_argument(
        "--map-seed",
        nargs="+",
        default=None,
        help="specific generator seed(s) to replay; if omitted, fresh random seeds are used",
    )
    parser.add_argument(
        "--generator",
        type=Path,
        default=DEFAULT_GENERATOR,
        help=f"official generator binary, default: {DEFAULT_GENERATOR}",
    )
    parser.add_argument(
        "--generator-timeout",
        type=float,
        default=DEFAULT_GENERATOR_TIMEOUT,
        help="timeout for one generator run",
    )
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for the solution")
    parser.add_argument("--runs-per-map", type=int, default=1, help="repeat each map with fresh repair randomness")
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=None,
        help="base seed for repair-point randomness; if omitted, each run uses a fresh random seed",
    )
    parser.add_argument("--best-total", type=float, default=DEFAULT_BEST_TOTAL, help="normalization baseline")
    parser.add_argument(
        "--official-tests",
        type=int,
        default=DEFAULT_OFFICIAL_TESTS,
        help="number of official tests used for score scaling",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="per-line interaction timeout")
    parser.add_argument(
        "--allow-eof-stop",
        action="store_true",
        help="treat clean EOF before the query limit as a valid early stop",
    )
    parser.add_argument("--show-stderr", action="store_true", help="print stderr produced by the solution")
    args = parser.parse_args()

    if not args.solution.is_file():
        parser.error(f"solution file not found: {args.solution}")
    if args.runs_per_map <= 0:
        parser.error("--runs-per-map must be positive")
    if args.generated_tests <= 0:
        parser.error("--generated-tests must be positive")
    if args.best_total <= 0:
        parser.error("--best-total must be positive")
    if args.official_tests <= 0:
        parser.error("--official-tests must be positive")
    if args.maps is not None and args.map_seed is not None:
        parser.error("--map-seed cannot be used together with --maps")

    cases: list[MapCase] = []
    if args.maps is not None:
        map_paths = expand_maps(args.maps)
        if not map_paths:
            parser.error("no maps selected")
        cases = [parse_map(path) for path in map_paths]
    else:
        ensure_generator(args.generator)
        map_seeds = args.map_seed if args.map_seed is not None else [fresh_map_seed() for _ in range(args.generated_tests)]
        for map_seed in map_seeds:
            cases.append(generate_map(args.generator, map_seed, args.generator_timeout))

    total = 0
    invalid = 0

    for case_index, case in enumerate(cases):
        for trial in range(1, args.runs_per_map + 1):
            if args.rng_seed is None:
                seed = fresh_repair_seed()
            else:
                seed = args.rng_seed + case_index * 1_000_003 + (trial - 1) * 97_409
            result = run_one(
                solution=args.solution,
                case=case,
                trial=trial,
                rng_seed=seed,
                python=args.python,
                timeout=args.timeout,
                allow_eof_stop=args.allow_eof_stop,
                show_stderr=args.show_stderr,
            )
            print_result(result)
            total += result.points
            invalid += result.status != "OK"

    runs = len(cases) * args.runs_per_map
    average_points = total / runs
    average_score = 800.0 * average_points * args.official_tests / args.best_total
    print()
    print(f"runs: {runs}")
    print(f"fresh_generated_maps: {args.maps is None}")
    print(f"invalid_runs: {invalid}")
    print(f"total_points: {total}")
    print(f"average_points: {average_points:.6f}")
    print(f"best_total_points: {args.best_total:g}")
    print(f"official_tests_for_scaling: {args.official_tests}")
    print(f"score: {average_score:.6f}")

    return 1 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
