# B. Lost Cursor

Source: <https://codeforces.com/gym/106478/problem/B>

Limits: 1 second, 256 MB.

This is an output-only / optimization problem. For each test, you need to provide a string of moves. The shorter the valid string is, the better the score.

## Story and Model

You are given a black-and-white `n x n` image, representing a monochrome screen. A cursor is located on one of the white pixels. Its initial position is unknown, but it is guaranteed to be white: if the cursor started on a black pixel, it would already be visible.

You may choose a fixed sequence of arrow-key presses:

- `U` - up;
- `D` - down;
- `L` - left;
- `R` - right.

After each key press, the cursor tries to move one cell in the chosen direction.

- If the move stays inside the image, the cursor moves to the neighboring pixel.
- If the move would leave the image, the cursor stays in place.
- If the cursor ever reaches a black pixel, it becomes visible, and that initial position is considered handled.

Your goal is to construct the shortest possible sequence of moves that is guaranteed to move the cursor onto a black pixel from every possible initial white pixel.

The sequence length must not exceed `5000`.

## Input

There is no standard input. Instead, the tests are provided in `problem-b-inputs.zip`.

The archive contains 8 tests:

- `01.png`
- `02.png`
- ...
- `08.png`

Each PNG is a grayscale image of size `n x n`, where `495 <= n <= 500`. Every pixel is either black or white. Each test contains at least one black pixel and at least one white pixel.

The downloaded tests are in [`inputs`](inputs).

## Output

Submit a zip archive containing files:

- `01.out`
- `02.out`
- ...
- `08.out`

You may omit some files.

Each `.out` file must contain a single line made only of the characters `U`, `D`, `L`, and `R`. This line is the move sequence for the corresponding PNG test.

## Scoring

For each test, the number of points of a solution is the length of its move string.

If the string does not guarantee reaching a black pixel from every possible initial white position, the score for that test is `0`.

If the string is valid, the final score for that test is:

```text
100 * (best_points / your_points)^2
```

where:

- `best_points` is the best known valid length among participants for this test;
- `your_points` is the length of your move string.

The scoreboard uses your best result for each test across all submissions.

## Examples

The archive `problem-b-samples.zip` contains two small samples. They are downloaded into [`samples`](samples).

### Sample 1

The image has size `2 x 2`. The only black pixel is at position `(2, 2)`.

Possible initial cursor positions:

- `(1, 1)`
- `(1, 2)`
- `(2, 1)`

One valid sequence is:

```text
RD
```

After `R`, the cursor starting from `(2, 1)` reaches `(2, 2)`. After the next move `D`, the other possible starting positions also reach `(2, 2)`.

### Sample 2

The image has size `4 x 4`.

One valid answer is:

```text
LLR
```

The illustrations in the Codeforces statement are enlarged and include a grid for clarity; the real sample files have exact sizes `2 x 2` and `4 x 4`.
