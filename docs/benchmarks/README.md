# Benchmark Baselines

Elio carries the shared GPU-triad benchmark runner because it is the compute
compiler and currently has the fewest historical benchmark artifacts.

Run from the Elio checkout with sibling `../selena` and `../eos` checkouts:

```sh
scripts/gpu-triad-bench.sh
```

The script writes one timestamped run under `docs/benchmarks/<run-id>/`:

```text
manifest.txt
elio.commit
elio.bench.jsonl
selena.commit
selena.bench.jsonl
eos.commit
eos.bench.jsonl
summary.md
summary.tsv
```

Each `*.bench.jsonl` file is raw `go test -json` output with benchmark lines
and allocation data. Keep the raw JSONL rather than a lossy summary so future
comparison tooling can choose p50, min, or other policy after a few runs exist.
The generated summaries use median values across repeated benchmark samples.

Useful knobs:

```sh
TRIAD_BENCH_COUNT=10 scripts/gpu-triad-bench.sh
TRIAD_BENCH_TIME=3s scripts/gpu-triad-bench.sh
TRIAD_BENCH_REGEX='Benchmark(Parse|Emit|Build)' scripts/gpu-triad-bench.sh
TRIAD_BENCH_REGEX=Benchmark scripts/gpu-triad-bench.sh
TRIAD_BENCH_OUT=/tmp/triad-bench scripts/gpu-triad-bench.sh
```

To summarize an existing run or compare it to a baseline:

```sh
go run ./scripts/gpu_triad_bench_summary.go docs/benchmarks/<run-id>
go run ./scripts/gpu_triad_bench_summary.go --baseline docs/benchmarks/<baseline-id> docs/benchmarks/<run-id>
go run ./scripts/gpu_triad_bench_summary.go --format tsv docs/benchmarks/<run-id>
```

The default regex is bounded to parser, emitter, CPU fallback, compile, and
build benchmarks so a baseline run does not accidentally invoke device or
long-running training benches. Set `TRIAD_BENCH_REGEX=Benchmark` when you
intentionally want every benchmark in all three repositories.

Tracked benchmark coverage starts with:

- Elio parser, text emitters, and CPU fallback execution.
- Selena parser and all-target compile path.
- Eos compiler build/artifact-size.

Refresh a baseline when a benchmark is added or an intentional performance
change lands. Do not refresh to hide ordinary run-to-run variance.
