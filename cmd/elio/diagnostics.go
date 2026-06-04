package main

import (
	"errors"
	"fmt"
	"io"
	"regexp"
	"strconv"
	"strings"

	"github.com/odvcencio/gotreesitter/taproot/diag"
)

type sourceError struct {
	file   string
	source []byte
	err    error
}

func (e *sourceError) Error() string { return e.err.Error() }
func (e *sourceError) Unwrap() error { return e.err }

func attachSource(file string, source []byte, err error) error {
	if err == nil {
		return nil
	}
	if len(diagnosticsFromError(err)) == 0 {
		return err
	}
	return &sourceError{file: file, source: source, err: err}
}

func printCommandError(w io.Writer, err error) {
	fmt.Fprintln(w, err)
	var se *sourceError
	if !errors.As(err, &se) {
		return
	}
	renderDiagnostics(w, se.file, se.source, se.err)
}

func renderDiagnostics(w io.Writer, file string, source []byte, err error) {
	for _, d := range diagnosticsFromError(err) {
		if d.Line > 0 {
			fmt.Fprintf(w, "  --> %s:%d:%d\n", file, d.Line, max(d.Col, 1))
		}
		fmt.Fprint(w, diag.Render(source, d))
	}
}

var lineColPattern = regexp.MustCompile(`^(.*?)([0-9]+):([0-9]+):\s*(.*)$`)

func diagnosticsFromError(err error) []diag.Diagnostic {
	if err == nil {
		return nil
	}
	var out []diag.Diagnostic
	for _, line := range strings.Split(err.Error(), "\n") {
		if d, ok := diagnosticFromLine(line); ok {
			out = append(out, d)
		}
	}
	return out
}

func diagnosticFromLine(line string) (diag.Diagnostic, bool) {
	m := lineColPattern.FindStringSubmatch(line)
	if m == nil {
		return diag.Diagnostic{}, false
	}
	lineNo, err := strconv.Atoi(m[2])
	if err != nil {
		return diag.Diagnostic{}, false
	}
	col, err := strconv.Atoi(m[3])
	if err != nil {
		return diag.Diagnostic{}, false
	}

	prefix := strings.TrimSpace(strings.TrimSuffix(m[1], ":"))
	message := strings.TrimSpace(m[4])
	code := "ELIO0001"
	hint := "Check the surrounding braces, declarations, and expression syntax."
	if strings.HasPrefix(prefix, "sema") {
		code = "ELIO1001"
		hint = "Check name resolution, mutability, type names, and builtin arity."
		prefix = strings.TrimSpace(strings.TrimPrefix(prefix, "sema:"))
	}
	if prefix != "" {
		message = prefix + ": " + message
	}
	return diag.Diagnostic{
		Code:     code,
		Severity: diag.SeverityError,
		Line:     lineNo,
		Col:      col,
		Message:  message,
		Hint:     hint,
	}, true
}
