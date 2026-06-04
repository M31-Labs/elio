#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
triad_root="${TRIAD_ROOT:-$(cd "$script_dir/../.." && pwd)}"
run_id="${TRIAD_BENCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
out_dir="${TRIAD_BENCH_OUT:-$script_dir/../docs/benchmarks/$run_id}"
bench_regex="${TRIAD_BENCH_REGEX:-Benchmark(Parse|Emit|RunCull|Compile|Build)}"
bench_count="${TRIAD_BENCH_COUNT:-5}"
bench_time="${TRIAD_BENCH_TIME:-1s}"

mkdir -p "$out_dir"

cat >"$out_dir/manifest.txt" <<EOF
run_id=$run_id
created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
triad_root=$triad_root
bench_regex=$bench_regex
bench_count=$bench_count
bench_time=$bench_time
go=$(go version)
EOF

for repo in elio selena eos; do
  repo_dir="$triad_root/$repo"
  if [[ ! -d "$repo_dir" ]]; then
    echo "missing repo: $repo_dir" >&2
    exit 1
  fi
  echo "==> $repo"
  (
    cd "$repo_dir"
    git rev-parse HEAD >"$out_dir/$repo.commit"
    go test -run '^$' -bench "$bench_regex" -benchmem -benchtime "$bench_time" -count "$bench_count" -json ./...
  ) | tee "$out_dir/$repo.bench.jsonl"
done

echo "wrote triad benchmark run: $out_dir"
