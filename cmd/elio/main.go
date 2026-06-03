// Command elio is the Elio compiler CLI: parse a .elio compute source and emit
// a backend shader, or check that it parses.
//
//	elio emit <wgsl|glsl|metal> <file.elio>   # emit shader source to stdout
//	elio check <file.elio>                    # parse-check; prints "ok" or errors
package main

import (
	"fmt"
	"io"
	"os"

	"m31labs.dev/elio/emit/glsl"
	"m31labs.dev/elio/emit/metal"
	"m31labs.dev/elio/emit/wgsl"
	"m31labs.dev/elio/ir"
	"m31labs.dev/elio/parse"
	"m31labs.dev/elio/sema"
)

const usage = `usage:
  elio emit <wgsl|glsl|metal> <file.elio>   emit shader source to stdout
  elio check <file.elio>                    parse + semantic-check the source`

func main() { os.Exit(run(os.Args[1:], os.Stdout, os.Stderr)) }

func run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		fmt.Fprintln(stderr, usage)
		return 2
	}
	switch args[0] {
	case "emit":
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
			fmt.Fprintln(stderr, err)
			return 1
		}
		fmt.Fprintln(stdout, "ok")
		return 0
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
		return nil, err
	}
	if err := sema.Errors(sema.Check(mod)); err != nil {
		return nil, err
	}
	return mod, nil
}

func emit(target, path string, stdout, stderr io.Writer) int {
	mod, err := loadModule(path)
	if err != nil {
		fmt.Fprintln(stderr, err)
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
		fmt.Fprintln(stderr, err)
		return 1
	}
	fmt.Fprint(stdout, out)
	return 0
}
