# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

**evolve** is an evolutionary search tool for Conway's Game of Life (B3/S23 rule). It uses a mutation-selection loop to discover initial cellular automaton configurations with exceptionally long lifespans before stabilizing. Output patterns are saved as `.rle` files readable by Golly and other Life viewers.

## Running the Search

```bash
# Run with the current active configuration
./doit

# Run directly with custom parameters
python3 gliders.py --checkpoint 300 --blank --p16 --spacex 9 --spacey 9 --nspark 100 --irad 1 --ispark 3 --rad 5 --prune 1000 --memory 40000
```

`doit` is a bash script containing the currently active command on line 2, plus a history of commented-out alternative configurations. Edit line 2 to change the search strategy.

## Visualizing Results

```bash
python3 chart.py --log results/log.<timestamp>
```

Parses `BEST`/`LMAX`/`BACK`/`ATH` lines from a run log and plots lifespan, population, and other metrics over time.

## Docker (for cluster runs)

```bash
cd docker
make build        # Build the 'life' Docker image
docker/launch     # Run the container with X11 forwarding and home directory mounted
```

The Makefile also has a `pull` target for distributing the image across a cluster of hydra-* machines.

## Architecture

### `gliders.py` — Main Evolutionary Engine

**Pattern representation**: A pattern is a dict `vec` mapping `(x, y)` integer coordinates to a *state index* into the global `gliders` list. Each state is a small collection of ON cells (a blinker, block, glider, r-pentomino, etc. in a specific orientation). The `render(pat, vec)` function writes these states onto a `lifelib` pattern object by placing each primitive at `(x*spacex, y*spacey)`.

**Lifespan measurement** (`lifespan()`): Runs the pattern forward in steps of `period=100`, computing the Shannon entropy of the population histogram over that window. When entropy drops below `0.8 * log(period)`, the pattern is considered stable and the generation count is returned. Returns `-1` for patterns that never stabilize ("runaways"). Each window is evaluated with a single call to `pat.advance_trace(step, period)` — a C++ loop that collects `period` population samples and returns the final pattern in one round-trip, replacing the previous inner Python loop (~4× speedup).

**Evolutionary pool** (`tree`): A list of `[lifespan, vec, failure_count]` triples. Each iteration:
1. Sample a parent uniformly (or weighted by lifespan with `--weighted`)
2. Mutate: apply `m ~ Exponential(1)` mutations, each adding a new randomly-placed state or (with `--delete`) removing an existing one
3. Evaluate lifespan of the mutant
4. If lifespan > parent's lifespan → append to pool, prune excess entries by removing (a) highest failure-count and (b) lowest-lifespan members
5. If lifespan ≤ parent's → increment parent's failure count

**Glider types** (flags that populate the `gliders` list): `--glider`, `--blinker`, `--block`, `--rpent`, `--meth`, `--pihep`, `--tub`, `--ttet`, `--bigblock`, `--figeight`, `--traffic`, `--p16`, `--blank`. Each flag appends the pattern and its symmetry-group variants.

**Radius algorithm** (`--radalgo`): Controls the spread of new mutations:
- `const`: fixed radius `--rad`
- `pi`: `sqrt(mean_lifespan) / π / spacex` (adaptive)

**Output files** in `results/`:
- `ATH_P{pop}_L{life}_seed{seed}_n{iter}.rle` — all-time high lifespan
- `BEST_P{pop}_L{life}_seed{seed}_n{iter}.rle` — best found at each `--checkpoint` interval
- `lmax_P{pop}_L{life}_seed{seed}_n{iter}.rle` — each new local maximum
- `init_K{k}_seed{seed}.rle` — first 10 initial seed patterns
- `log.<timestamp>` — full run log parsed by `chart.py`

### Key CLI Parameters

| Parameter | Meaning |
|-----------|---------|
| `--memory` | `lifelib` memory limit in MB |
| `--prune` | Max pool size (prune when exceeded) |
| `--nspark` | Number of initial patterns to seed the pool |
| `--ispark` | Max number of primitives per initial pattern |
| `--irad` | Uniform radius for placing initial sparks |
| `--rad` | Gaussian radius for mutation displacement |
| `--spacex/y` | Grid spacing between pattern primitives |
| `--checkpoint` | Seconds between saving BEST snapshots |
| `--seed` | Random seed (auto-generated if omitted) |

### lifelib dependency

`gliders.py` depends on a patched build of lifelib. The upstream source is at `https://gitlab.com/apgoucher/lifelib`; the local fork lives at `~/lifelib` (git repo, branch `master`, commit `b17974b`).

Two files are patched beyond upstream:

- **`lifelib.cpp`**: adds `AdvancePatternWithTrace(void* pat, int step, int period, uint64_t* pops)` — advances the pattern `period` times by `step` generations each, recording `popcount(1000000007)` into `pops[0..period-1]` and storing the resulting `apg::pattern*` as `pops[period]`.
- **`pythlib/pattern.py`**: adds `Pattern.advance_trace(step=1, period=100)` — calls `AdvancePatternWithTrace` via the existing ctypes dispatch, returns `(new_pattern, uint64_numpy_array)`.

To rebuild and redeploy after modifying `~/lifelib`:

```bash
cd ~/lifelib
g++ -std=c++11 -march=native -O3 -g -Wall -Wextra -fdiagnostics-color=always \
    -fPIC -shared -o pythlib/lifelib_b3s23.so lifelib.cpp
SITE=~/anaconda3/lib/python3.11/site-packages/lifelib
cp pythlib/lifelib_b3s23.so $SITE/pythlib/lifelib_b3s23.so
cp pythlib/pattern.py       $SITE/pythlib/pattern.py
cp lifelib.cpp              $SITE/lifelib.cpp
```

The `avxlife/lifelogic/` headers (generated, not committed) must be present before compiling. If missing, run `python3 -c "import lifelib; lifelib.compile_rules('b3s23', force_compile=True)"` once to regenerate them.

### Version History

`gliders.py.1` through `gliders.py.6` are snapshots of prior versions; `OLD/` contains even earlier iterations (`entropy.py.*`, `hello.py.*`). `r1/` through `r16/` are results directories from individual runs.
