package ts

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"m31labs.dev/elio/graph"
)

// graphTypes is the ES module's type for a graph (the classic-script form
// documents it in JSDoc).
const graphTypes = `
/** A pass graph; see the Elio pass graph contract. */
export interface ElioGraph {
  name: string;
  backend: "webgpu" | "webgl2";
  tier: "high" | "medium" | "low";
  version: number;
  params: { name: string; type: string; count?: number; tween?: number; source: Record<string, unknown> }[];
  resources: Record<string, unknown>[];
  passes: Record<string, unknown>[];
  frameSwaps: [string, string][];
  exports: Record<string, string>;
}
`

// graphsLiteral validates the set's graphs and renders them as one JSON
// object keyed by NamedGraph.Key, or "" when the set has none.
func graphsLiteral(set KernelSet) (string, error) {
	if len(set.Graphs) == 0 {
		return "", nil
	}
	ks := set.GraphKernels()
	var errs []error
	seen := map[string]bool{}
	var b strings.Builder
	b.WriteString("{\n")
	for i, ng := range set.Graphs {
		g := ng.Graph
		if want := g.Backend + "/" + g.Tier; ng.Key != want || seen[ng.Key] {
			errs = append(errs, fmt.Errorf("ts: graph key %q must be unique and equal %q", ng.Key, want))
		}
		seen[ng.Key] = true
		if err := graph.Validate(g, ks); err != nil {
			errs = append(errs, fmt.Errorf("ts: graph %q: %w", ng.Key, err))
		}
		sep := ","
		if i == len(set.Graphs)-1 {
			sep = ""
		}
		fmt.Fprintf(&b, "  %s: %s%s\n", quote(ng.Key), compactLit(g), sep)
	}
	b.WriteString("}")
	return b.String(), errors.Join(errs...)
}

// compactLit renders v as one line of JSON. Graphs are large and repetitive
// (the WebGL2 FFT alone is over a hundred passes), so they are not indented.
func compactLit(v any) string {
	var b bytes.Buffer
	enc := json.NewEncoder(&b)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		panic(err) // plain structs only
	}
	return strings.TrimRight(b.String(), "\n")
}

// indent prefixes every line after the first with pad.
func indent(s, pad string) string { return strings.ReplaceAll(s, "\n", "\n"+pad) }
