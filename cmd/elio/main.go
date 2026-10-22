// Command elio is the Elio compiler CLI: parse a .elio compute source and emit
// a backend shader, or check that it parses.
//
//	elio emit <wgsl|glsl|metal> <file.elio>   # emit shader source to stdout
//	elio emit ts --set ocean                  # TypeScript module of the ocean kernels
//	elio check <file.elio>                    # parse-check; prints "ok" or errors
package main

import (
	"fmt"
	"io"
	"os"

	"m31labs.dev/elio/emit/glsl"
	"m31labs.dev/elio/emit/metal"
	"m31labs.dev/elio/emit/ts"
	"m31labs.dev/elio/emit/wgsl"
	"m31labs.dev/elio/ir"
	"m31labs.dev/elio/parse"
	"m31labs.dev/elio/sema"
	"m31labs.dev/elio/stdlib"
)

// version names this Elio build in generated headers.
const version = "v0.4.3+ocean"

func runEmitTS(args []string, stdout, stderr io.Writer) int {
	if len(args) != 2 || args[0] != "--set" {
		fmt.Fprintln(stderr, "usage: elio emit ts --set ocean")
		return 2
	}
	var set ts.KernelSet
	var err error
	switch args[1] {
	case "ocean":
		set, err = stdlib.OceanKernelSet()
	default:
		fmt.Fprintf(stderr, "elio: unknown kernel set %q (want ocean)\n", args[1])
		return 2
	}
	if err == nil {
		var out string
		if out, err = ts.Emit(set, ts.Meta{Version: version}); err == nil {
			fmt.Fprint(stdout, out)
			return 0
		}
	}
	printCommandError(stderr, err)
	return 1
}

const usage = `usage:
  elio emit <wgsl|glsl|metal> <file.elio>   emit shader source to stdout
  elio emit ts --set ocean                  emit a TypeScript module of a kernel set
  elio check <file.elio>                    parse + semantic-check the source
  elio graph [--format json|dot] <file.elio>
  elio kernels [--target wgsl|glsl|metal] [--out dir] [--validate] <file.elio>
  elio compile --bundle dir [--validate-kernels] <file.elio>
  elio doctor`

func main() { os.Exit(run(os.Args[1:], os.Stdout, os.Stderr)) }

func run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, usage)
		return 2
	}
	switch args[0] {
	case "emit":
		if len(args) >= 2 && args[1] == "ts" {
			return runEmitTS(args[2:], stdout, stderr)
		}
		if len(args) != 3 {
			fmt.Fprintln(stderr, "usage: elio emit <wgsl|glsl|metal> <file.elio>")
			return 2
		}
		return emit(args[1], args[2], stdout, stderr)
	case "check":
		if len(args) != 2 {
			fmt.Fprintln(stderr, "usage: elio check <file.elio>")
			return 2
		}
		if _, err := loadModule(args[1]); err != nil {
			printCommandError(stderr, err)
			return 1
		}
		fmt.Fprintln(stdout, "ok")
		return 0
	case "graph":
		return runGraph(args[1:], stdout, stderr)
	case "kernels":
		return runKernels(args[1:], stdout, stderr)
	case "compile":
		return runCompile(args[1:], stdout, stderr)
	case "doctor":
		return runDoctor(args[1:], stdout, stderr)
	default:
		fmt.Fprintln(stderr, usage)
		return 2
	}
}

// loadModule parses path and runs the semantic checker, so both `check` and
// `emit` reject invalid source before any backend sees it.
func loadModule(path string) (*ir.Module, error) {
	src, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	mod, err := parse.Parse(string(src))
	if err != nil {
		return nil, attachSource(path, src, err)
	}
	if err := sema.Errors(sema.Check(mod)); err != nil {
		return nil, attachSource(path, src, err)
	}
	return mod, nil
}

func emit(target, path string, stdout, stderr io.Writer) int {
	mod, err := loadModule(path)
	if err != nil {
		printCommandError(stderr, err)
		return 1
	}
	var out string
	switch target {
	case "wgsl":
		out, err = wgsl.Emit(mod)
	case "glsl":
		out, err = glsl.Emit(mod)
	case "metal":
		out, err = metal.Emit(mod)
	default:
		fmt.Fprintf(stderr, "elio: unknown target %q (want wgsl|glsl|metal)\n", target)
		return 2
	}
	if err != nil {
		printCommandError(stderr, err)
		return 1
	}
	fmt.Fprint(stdout, out)
	return 0
}
