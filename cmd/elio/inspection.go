package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	goruntime "runtime"
	"sort"
	"strings"
	"time"

	"m31labs.dev/elio/emit/glsl"
	"m31labs.dev/elio/emit/metal"
	"m31labs.dev/elio/emit/wgsl"
	"m31labs.dev/elio/ir"
	prismvalidate "m31labs.dev/prism/validate"
)

var inspectionTargets = []string{"wgsl", "glsl", "metal"}

type graphReport struct {
	GraphVersion int            `json:"graph_version"`
	SourcePath   string         `json:"source_path"`
	Counts       graphCounts    `json:"counts"`
	Module       moduleSnapshot `json:"module"`
}

type graphCounts struct {
	Structs  int `json:"structs"`
	Consts   int `json:"consts"`
	Bindings int `json:"bindings"`
	Kernels  int `json:"kernels"`
}

type moduleSnapshot struct {
	Structs  []ir.Struct      `json:"structs,omitempty"`
	Consts   []constSnapshot  `json:"consts,omitempty"`
	Bindings []ir.Binding     `json:"bindings,omitempty"`
	Kernels  []kernelSnapshot `json:"kernels,omitempty"`
}

type constSnapshot struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

type kernelSnapshot struct {
	Name          string       `json:"name"`
	WorkgroupSize [3]int       `json:"workgroup_size"`
	Builtins      []ir.Builtin `json:"builtins,omitempty"`
	Shared        []ir.Shared  `json:"shared,omitempty"`
	Statements    int          `json:"statements"`
}

func runGraph(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("graph", flag.ContinueOnError)
	fs.SetOutput(stderr)
	format := fs.String("format", "json", "output format: json or dot")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "usage: elio graph [--format json|dot] <file.elio>")
		return 2
	}
	mod, err := loadModule(fs.Arg(0))
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	report := newGraphReport(fs.Arg(0), mod)
	switch *format {
	case "json":
		if err := writeJSON(stdout, report); err != nil {
			fmt.Fprintln(stderr, err)
			return 1
		}
	case "dot":
		fmt.Fprint(stdout, graphDOT(report))
	default:
		fmt.Fprintf(stderr, "unsupported graph format %q\n", *format)
		return 2
	}
	return 0
}

func newGraphReport(path string, mod *ir.Module) graphReport {
	report := graphReport{GraphVersion: 1, SourcePath: path}
	if mod == nil {
		return report
	}
	report.Counts = graphCounts{
		Structs:  len(mod.Structs),
		Consts:   len(mod.Consts),
		Bindings: len(mod.Bindings),
		Kernels:  len(mod.Kernels),
	}
	report.Module.Structs = mod.Structs
	report.Module.Bindings = mod.Bindings
	for _, c := range mod.Consts {
		report.Module.Consts = append(report.Module.Consts, constSnapshot{Name: c.Name, Type: typeSummary(c.Type)})
	}
	for _, k := range mod.Kernels {
		report.Module.Kernels = append(report.Module.Kernels, kernelSnapshot{
			Name:          k.Name,
			WorkgroupSize: k.WorkgroupSize,
			Builtins:      k.Builtins,
			Shared:        k.Shared,
			Statements:    len(k.Body),
		})
	}
	return report
}

func graphDOT(report graphReport) string {
	var b strings.Builder
	b.WriteString("digraph elio {\n")
	b.WriteString("  module [label=\"module\", shape=box];\n")
	for _, kernel := range report.Module.Kernels {
		name := sanitizeIdent(kernel.Name)
		fmt.Fprintf(&b, "  kernel_%s [label=%q, shape=component];\n", name, "kernel "+kernel.Name)
		fmt.Fprintf(&b, "  module -> kernel_%s;\n", name)
	}
	b.WriteString("}\n")
	return b.String()
}

type targetSourceManifest struct {
	ManifestVersion int                 `json:"manifest_version"`
	CreatedAt       string              `json:"created_at,omitempty"`
	SourcePath      string              `json:"source_path,omitempty"`
	Module          string              `json:"module"`
	Targets         []string            `json:"targets"`
	SourceCount     int                 `json:"source_count"`
	Sources         []targetSourceEntry `json:"sources"`
}

type targetSourceEntry struct {
	Target      string            `json:"target"`
	SourceFile  string            `json:"source_file"`
	SourceBytes int               `json:"source_bytes"`
	Kernels     []entryValidation `json:"kernels,omitempty"`
}

type entryValidation struct {
	Entry         string `json:"entry"`
	EntryChecked  bool   `json:"entry_checked,omitempty"`
	ToolSkipped   bool   `json:"tool_skipped,omitempty"`
	ToolError     string `json:"tool_error,omitempty"`
	ToolOutputLen int    `json:"tool_output_len,omitempty"`
}

func runKernels(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("kernels", flag.ContinueOnError)
	fs.SetOutput(stderr)
	targetFilter := fs.String("target", "", "target to extract; empty extracts all")
	outDir := fs.String("out", "kernels", "directory for extracted target sources")
	validateSources := fs.Bool("validate", false, "record Prism source validation status")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if fs.NArg() != 1 {
		fmt.Fprintln(stderr, "usage: elio kernels [--target wgsl|glsl|metal] [--out dir] [--validate] <file.elio>")
		return 2
	}
	mod, err := loadModule(fs.Arg(0))
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	manifest, err := writeTargetSources(mod, *outDir, *targetFilter, "", *validateSources)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	manifest.SourcePath = fs.Arg(0)
	if err := writeJSONFile(filepath.Join(*outDir, "manifest.json"), manifest); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	fmt.Fprintf(stdout, "wrote %d target source(s) -> %s\n", manifest.SourceCount, *outDir)
	return 0
}

func runCompile(args []string, stdout, stderr io.Writer) int {
	fs := flag.NewFlagSet("compile", flag.ContinueOnError)
	fs.SetOutput(stderr)
	bundleDir := fs.String("bundle", "", "write inspection bundle sidecar directory")
	validateSources := fs.Bool("validate-kernels", false, "record Prism source validation status in the bundle manifest")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *bundleDir == "" || fs.NArg() != 1 {
		fmt.Fprintln(stderr, "usage: elio compile --bundle dir [--validate-kernels] <file.elio>")
		return 2
	}
	srcPath := fs.Arg(0)
	src, err := os.ReadFile(srcPath)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	mod, err := loadModule(srcPath)
	if err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	if err := writeCompileBundle(*bundleDir, srcPath, src, mod, *validateSources); err != nil {
		fmt.Fprintln(stderr, err)
		return 1
	}
	fmt.Fprintf(stdout, "bundle: %s\n", *bundleDir)
	return 0
}

func writeCompileBundle(dir, srcPath string, src []byte, mod *ir.Module, validateSources bool) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "source.elio"), src, 0o644); err != nil {
		return err
	}
	if err := writeJSONFile(filepath.Join(dir, "graph.json"), newGraphReport(srcPath, mod)); err != nil {
		return err
	}
	sourceManifest, err := writeTargetSources(mod, filepath.Join(dir, "kernels"), "", "kernels", validateSources)
	if err != nil {
		return err
	}
	manifest := struct {
		BundleVersion int                 `json:"bundle_version"`
		CreatedAt     string              `json:"created_at"`
		SourcePath    string              `json:"source_path"`
		Kernels       int                 `json:"kernels"`
		Targets       []string            `json:"targets"`
		SourceCount   int                 `json:"source_count"`
		Sources       []targetSourceEntry `json:"sources"`
	}{
		BundleVersion: 1,
		CreatedAt:     time.Now().UTC().Format(time.RFC3339),
		SourcePath:    srcPath,
		Kernels:       len(mod.Kernels),
		Targets:       sourceManifest.Targets,
		SourceCount:   sourceManifest.SourceCount,
		Sources:       sourceManifest.Sources,
	}
	return writeJSONFile(filepath.Join(dir, "manifest.json"), manifest)
}

func writeTargetSources(mod *ir.Module, outDir, targetFilter, manifestPrefix string, validateSources bool) (targetSourceManifest, error) {
	if mod == nil {
		return targetSourceManifest{}, fmt.Errorf("module is nil")
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return targetSourceManifest{}, err
	}
	targets, err := selectedTargets(targetFilter)
	if err != nil {
		return targetSourceManifest{}, err
	}
	manifest := targetSourceManifest{
		ManifestVersion: 1,
		CreatedAt:       time.Now().UTC().Format(time.RFC3339),
		Module:          moduleName(mod),
		Targets:         append([]string(nil), targets...),
	}
	sort.Strings(manifest.Targets)
	for _, target := range targets {
		source, err := emitTargetSource(target, mod)
		if err != nil {
			return targetSourceManifest{}, err
		}
		filename := "module." + targetSourceExt(target)
		if targetFilter == "" {
			filename = target + "." + targetSourceExt(target)
		}
		if err := os.WriteFile(filepath.Join(outDir, filename), []byte(source), 0o644); err != nil {
			return targetSourceManifest{}, err
		}
		sourceFile := filename
		if manifestPrefix != "" {
			sourceFile = filepath.ToSlash(filepath.Join(manifestPrefix, filename))
		}
		entry := targetSourceEntry{
			Target:      target,
			SourceFile:  sourceFile,
			SourceBytes: len(source),
		}
		if validateSources {
			entry.Kernels = validateTargetEntries(target, source, mod.Kernels)
		}
		manifest.Sources = append(manifest.Sources, entry)
		manifest.SourceCount++
	}
	return manifest, nil
}

func selectedTargets(filter string) ([]string, error) {
	if filter == "" {
		return append([]string(nil), inspectionTargets...), nil
	}
	for _, target := range inspectionTargets {
		if filter == target {
			return []string{target}, nil
		}
	}
	return nil, fmt.Errorf("unknown target %q (want wgsl|glsl|metal)", filter)
}

func emitTargetSource(target string, mod *ir.Module) (string, error) {
	switch target {
	case "wgsl":
		return wgsl.Emit(mod)
	case "glsl":
		return glsl.Emit(mod)
	case "metal":
		return metal.Emit(mod)
	default:
		return "", fmt.Errorf("unknown target %q", target)
	}
}

func validateTargetEntries(target, source string, kernels []ir.Kernel) []entryValidation {
	out := make([]entryValidation, 0, len(kernels))
	for _, kernel := range kernels {
		entry := validationEntry(target, kernel.Name)
		v := entryValidation{Entry: entry}
		src := prismvalidate.Source{
			Name:    kernel.Name,
			Backend: validationBackend(target),
			Entry:   entry,
			Source:  source,
		}
		if err := prismvalidate.CheckSource(src); err != nil {
			v.ToolError = err.Error()
			out = append(out, v)
			continue
		}
		v.EntryChecked = true
		res, err := prismvalidate.RunSource(src)
		v.ToolSkipped = res.Skipped
		v.ToolOutputLen = len(res.Output)
		if err != nil {
			v.ToolError = err.Error()
		}
		out = append(out, v)
	}
	return out
}

func validationBackend(target string) prismvalidate.Backend {
	switch target {
	case "wgsl":
		return prismvalidate.BackendWebGPU
	case "glsl":
		return prismvalidate.BackendVulkan
	case "metal":
		return prismvalidate.BackendMetal
	default:
		return ""
	}
}

func validationEntry(target, kernel string) string {
	if target == "glsl" {
		return "main"
	}
	return kernel
}

func targetSourceExt(target string) string {
	switch target {
	case "wgsl":
		return "wgsl"
	case "glsl":
		return "comp"
	case "metal":
		return "metal"
	default:
		return "txt"
	}
}

func moduleName(mod *ir.Module) string {
	if mod == nil || len(mod.Kernels) == 0 {
		return "module"
	}
	return mod.Kernels[0].Name
}

func runDoctor(args []string, stdout, stderr io.Writer) int {
	if len(args) != 0 {
		fmt.Fprintln(stderr, "usage: elio doctor")
		return 2
	}
	fmt.Fprintln(stdout, "elio: dev")
	fmt.Fprintf(stdout, "go: %s %s/%s\n", goruntime.Version(), goruntime.GOOS, goruntime.GOARCH)
	fmt.Fprintln(stdout, "targets: wgsl glsl metal")
	fmt.Fprintln(stdout, "tools:")
	for _, tool := range []string{"naga", "glslangValidator", "xcrun"} {
		if path, err := exec.LookPath(tool); err == nil {
			fmt.Fprintf(stdout, "  %s: %s\n", tool, path)
		} else {
			fmt.Fprintf(stdout, "  %s: unavailable\n", tool)
		}
	}
	return 0
}

func writeJSONFile(path string, value any) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	return writeJSON(file, value)
}

func writeJSON(w io.Writer, value any) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(value)
}

func sanitizeIdent(value string) string {
	if value == "" {
		return "unnamed"
	}
	var b strings.Builder
	for i, r := range value {
		if r == '_' || (r >= 'A' && r <= 'Z') || (r >= 'a' && r <= 'z') || (i > 0 && r >= '0' && r <= '9') {
			b.WriteRune(r)
			continue
		}
		b.WriteByte('_')
	}
	return b.String()
}

func typeSummary(t ir.Type) string {
	if t == nil {
		return ""
	}
	return fmt.Sprintf("%T", t)
}
