package main

import (
	"strings"
	"testing"
)

func TestParseBenchmarkLine(t *testing.T) {
	line := "BenchmarkEmitCull-16    12345    678.9 ns/op    64 B/op    2 allocs/op    4096 source_bytes\n"
	s, ok := parseBenchmarkLine("elio", "m31labs.dev/elio/emit/wgsl", line)
	if !ok {
		t.Fatal("benchmark line was not parsed")
	}
	if s.Name != "BenchmarkEmitCull" || s.Metrics["ns/op"] != 678.9 || s.Metrics["source_bytes"] != 4096 {
		t.Fatalf("unexpected sample: %+v", s)
	}
}

func TestSummarizeUsesMedian(t *testing.T) {
	summaries := summarize([]sample{
		{Repo: "elio", Package: "pkg", Name: "BenchmarkX", Metrics: map[string]float64{"ns/op": 30}},
		{Repo: "elio", Package: "pkg", Name: "BenchmarkX", Metrics: map[string]float64{"ns/op": 10}},
		{Repo: "elio", Package: "pkg", Name: "BenchmarkX", Metrics: map[string]float64{"ns/op": 20}},
	})
	got := summaries[key("elio", "pkg", "BenchmarkX")]
	if got.Metrics["ns/op"] != 20 || got.Runs != 3 {
		t.Fatalf("summary = %+v", got)
	}
}

func TestWriteMarkdownWithBaselineDelta(t *testing.T) {
	current := map[string]summary{
		key("elio", "pkg", "BenchmarkX"): {Repo: "elio", Package: "pkg", Name: "BenchmarkX", Runs: 1, Metrics: map[string]float64{"ns/op": 110}},
	}
	baseline := map[string]summary{
		key("elio", "pkg", "BenchmarkX"): {Repo: "elio", Package: "pkg", Name: "BenchmarkX", Runs: 1, Metrics: map[string]float64{"ns/op": 100}},
	}
	var out strings.Builder
	writeMarkdown(&out, current, baseline)
	if !strings.Contains(out.String(), "+10.0%") {
		t.Fatalf("markdown missing delta:\n%s", out.String())
	}
}
