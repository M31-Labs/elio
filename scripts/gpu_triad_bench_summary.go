package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

type event struct {
	Action  string `json:"Action"`
	Package string `json:"Package"`
	Output  string `json:"Output"`
}

type sample struct {
	Repo    string
	Package string
	Name    string
	Metrics map[string]float64
}

type summary struct {
	Repo    string
	Package string
	Name    string
	Metrics map[string]float64
	Runs    int
}

var benchNameCPU = regexp.MustCompile(`-\d+$`)

func main() {
	format := flag.String("format", "markdown", "output format: markdown or tsv")
	baselineDir := flag.String("baseline", "", "optional baseline benchmark run directory")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: go run ./scripts/gpu_triad_bench_summary.go [--format markdown|tsv] [--baseline dir] <run-dir>")
		os.Exit(2)
	}
	current, err := readRun(flag.Arg(0))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	var baseline map[string]summary
	if *baselineDir != "" {
		baseline, err = readRun(*baselineDir)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
	switch *format {
	case "markdown":
		writeMarkdown(os.Stdout, current, baseline)
	case "tsv":
		writeTSV(os.Stdout, current, baseline)
	default:
		fmt.Fprintf(os.Stderr, "unsupported format %q\n", *format)
		os.Exit(2)
	}
}

func readRun(dir string) (map[string]summary, error) {
	matches, err := filepath.Glob(filepath.Join(dir, "*.bench.jsonl"))
	if err != nil {
		return nil, err
	}
	if len(matches) == 0 {
		return nil, fmt.Errorf("no *.bench.jsonl files found in %s", dir)
	}
	var samples []sample
	for _, path := range matches {
		repo := strings.TrimSuffix(filepath.Base(path), ".bench.jsonl")
		file, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		next, readErr := readSamples(repo, file)
		closeErr := file.Close()
		if readErr != nil {
			return nil, fmt.Errorf("%s: %w", path, readErr)
		}
		if closeErr != nil {
			return nil, closeErr
		}
		samples = append(samples, next...)
	}
	return summarize(samples), nil
}

func readSamples(repo string, r io.Reader) ([]sample, error) {
	scanner := bufio.NewScanner(r)
	var out []sample
	for scanner.Scan() {
		var ev event
		if err := json.Unmarshal(scanner.Bytes(), &ev); err != nil {
			return nil, err
		}
		if ev.Action != "output" || ev.Output == "" {
			continue
		}
		if s, ok := parseBenchmarkLine(repo, ev.Package, ev.Output); ok {
			out = append(out, s)
		}
	}
	return out, scanner.Err()
}

func parseBenchmarkLine(repo, pkg, line string) (sample, bool) {
	fields := strings.Fields(line)
	if len(fields) < 4 || !strings.HasPrefix(fields[0], "Benchmark") {
		return sample{}, false
	}
	name := benchNameCPU.ReplaceAllString(fields[0], "")
	metrics := map[string]float64{}
	if n, err := strconv.ParseFloat(fields[1], 64); err == nil {
		metrics["runs/op"] = n
	}
	for i := 2; i+1 < len(fields); i += 2 {
		value, err := strconv.ParseFloat(fields[i], 64)
		if err != nil {
			continue
		}
		metrics[fields[i+1]] = value
	}
	return sample{Repo: repo, Package: pkg, Name: name, Metrics: metrics}, true
}

func summarize(samples []sample) map[string]summary {
	grouped := map[string][]sample{}
	for _, s := range samples {
		grouped[key(s.Repo, s.Package, s.Name)] = append(grouped[key(s.Repo, s.Package, s.Name)], s)
	}
	out := map[string]summary{}
	for k, group := range grouped {
		metricValues := map[string][]float64{}
		for _, s := range group {
			for metric, value := range s.Metrics {
				metricValues[metric] = append(metricValues[metric], value)
			}
		}
		metrics := map[string]float64{}
		for metric, values := range metricValues {
			metrics[metric] = median(values)
		}
		first := group[0]
		out[k] = summary{Repo: first.Repo, Package: first.Package, Name: first.Name, Metrics: metrics, Runs: len(group)}
	}
	return out
}

func median(values []float64) float64 {
	if len(values) == 0 {
		return math.NaN()
	}
	cp := append([]float64(nil), values...)
	sort.Float64s(cp)
	mid := len(cp) / 2
	if len(cp)%2 == 1 {
		return cp[mid]
	}
	return (cp[mid-1] + cp[mid]) / 2
}

func writeMarkdown(w io.Writer, current, baseline map[string]summary) {
	keys := sortedKeys(current)
	fmt.Fprintln(w, "| repo | package | benchmark | runs | ns/op | delta | B/op | allocs/op | source_bytes | artifact_bytes |")
	fmt.Fprintln(w, "|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
	for _, k := range keys {
		s := current[k]
		fmt.Fprintf(w, "| %s | %s | %s | %d | %s | %s | %s | %s | %s | %s |\n",
			s.Repo, shortPackage(s.Package), s.Name, s.Runs,
			formatMetric(s.Metrics["ns/op"]),
			formatDelta(s, baseline[k]),
			formatMetric(s.Metrics["B/op"]),
			formatMetric(s.Metrics["allocs/op"]),
			formatMetric(s.Metrics["source_bytes"]),
			formatMetric(s.Metrics["artifact_bytes"]))
	}
}

func writeTSV(w io.Writer, current, baseline map[string]summary) {
	fmt.Fprintln(w, "repo\tpackage\tbenchmark\truns\tns/op\tdelta_pct\tB/op\tallocs/op\tsource_bytes\tartifact_bytes")
	for _, k := range sortedKeys(current) {
		s := current[k]
		fmt.Fprintf(w, "%s\t%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n",
			s.Repo, s.Package, s.Name, s.Runs,
			rawMetric(s.Metrics["ns/op"]),
			rawDelta(s, baseline[k]),
			rawMetric(s.Metrics["B/op"]),
			rawMetric(s.Metrics["allocs/op"]),
			rawMetric(s.Metrics["source_bytes"]),
			rawMetric(s.Metrics["artifact_bytes"]))
	}
}

func sortedKeys(m map[string]summary) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		a, b := m[keys[i]], m[keys[j]]
		if a.Repo != b.Repo {
			return a.Repo < b.Repo
		}
		if a.Package != b.Package {
			return a.Package < b.Package
		}
		return a.Name < b.Name
	})
	return keys
}

func formatMetric(v float64) string {
	if v == 0 || math.IsNaN(v) {
		return "-"
	}
	if math.Abs(v) >= 100 {
		return fmt.Sprintf("%.0f", v)
	}
	return fmt.Sprintf("%.2f", v)
}

func rawMetric(v float64) string {
	if v == 0 || math.IsNaN(v) {
		return ""
	}
	return strconv.FormatFloat(v, 'f', -1, 64)
}

func formatDelta(current, baseline summary) string {
	raw := rawDelta(current, baseline)
	if raw == "" {
		return "-"
	}
	v, _ := strconv.ParseFloat(raw, 64)
	return fmt.Sprintf("%+.1f%%", v)
}

func rawDelta(current, baseline summary) string {
	cur, base := current.Metrics["ns/op"], baseline.Metrics["ns/op"]
	if cur == 0 || base == 0 || math.IsNaN(cur) || math.IsNaN(base) {
		return ""
	}
	return strconv.FormatFloat(((cur-base)/base)*100, 'f', 3, 64)
}

func shortPackage(pkg string) string {
	const prefix = "m31labs.dev/"
	return strings.TrimPrefix(pkg, prefix)
}

func key(repo, pkg, name string) string {
	return repo + "\x00" + pkg + "\x00" + name
}
