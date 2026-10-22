package graph

import (
	"errors"
	"fmt"
)

// Expand returns g with every repeat pass replaced by its iterations, in
// order. Each expanded pass is named "<repeat>[<i>].<pass>"; the index param is
// replaced by the iteration number in uniform values; and the resources of
// each swap pair are exchanged in the bindings that follow, so the result
// holds only plain passes with concrete resource names. g is not modified.
// Expand reports malformed repeats and swaps that are not undone by the end of
// the pass list.
func Expand(g Graph) (Graph, error) {
	e := &expander{res: map[string]Resource{}, alias: map[string]string{}, params: map[string]bool{}}
	for _, r := range g.Resources {
		e.res[r.Name] = r
	}
	for _, p := range g.Params {
		e.params[p.Name] = true
	}
	out := g
	out.Passes = e.walk(g.Passes, map[string]int{}, "", "")
	for name, to := range e.alias {
		if name != to {
			e.errf("swaps are not restored: %q still refers to %q at the end of the passes", name, to)
			break
		}
	}
	return out, errors.Join(e.errs...)
}

type expander struct {
	res    map[string]Resource
	alias  map[string]string // name -> the resource it currently refers to
	params map[string]bool
	errs   []error
}

func (e *expander) errf(format string, a ...any) {
	e.errs = append(e.errs, fmt.Errorf("graph: %s", fmt.Sprintf(format, a...)))
}

func (e *expander) resolve(name string) string {
	if to, ok := e.alias[name]; ok {
		return to
	}
	return name
}

func (e *expander) walk(ps []Pass, env map[string]int, prefix, when string) []Pass {
	var out []Pass
	for _, p := range ps {
		if p.Type != "repeat" {
			out = append(out, e.plain(p, env, prefix, when))
			continue
		}
		name := prefix + p.Name
		switch {
		case p.Count < 1:
			e.errf("repeat %q: count %d, want at least 1", name, p.Count)
			continue
		case p.Index == "" || e.params[p.Index]:
			e.errf("repeat %q: index %q is empty or names a declared param", name, p.Index)
			continue
		case len(p.Body) == 0:
			e.errf("repeat %q: body is empty", name)
			continue
		}
		if _, nested := env[p.Index]; nested {
			e.errf("repeat %q: index %q is already in use by an outer repeat", name, p.Index)
			continue
		}
		ok := true
		for _, s := range p.Swap {
			a, aok := e.res[s[0]]
			b, bok := e.res[s[1]]
			if !aok || !bok || !sameShape(a, b) {
				e.errf("repeat %q: swap %v names an unknown resource or two that differ in shape", name, s)
				ok = false
			}
		}
		if !ok {
			continue
		}
		inner := p.When
		if inner == "" {
			inner = when
		}
		for i := 0; i < p.Count; i++ {
			env2 := map[string]int{p.Index: i}
			for k, v := range env {
				env2[k] = v
			}
			out = append(out, e.walk(p.Body, env2, fmt.Sprintf("%s[%d].", name, i), inner)...)
			for _, s := range p.Swap {
				a, b := e.resolve(s[0]), e.resolve(s[1])
				e.alias[s[0]], e.alias[s[1]] = b, a
			}
		}
	}
	return out
}

func (e *expander) plain(p Pass, env map[string]int, prefix, when string) Pass {
	q := p
	q.Name = prefix + p.Name
	if when != "" && (p.When == "" || p.When == "always") {
		q.When = when
	}
	if p.Resource != "" {
		q.Resource = e.resolve(p.Resource)
	}
	if p.Bindings != nil {
		q.Bindings = make([]Binding, len(p.Bindings))
		for i, b := range p.Bindings {
			b.Resource = e.resolve(b.Resource)
			q.Bindings[i] = b
		}
	}
	if p.Uniforms != nil {
		q.Uniforms = make([]Uniform, len(p.Uniforms))
		for i, u := range p.Uniforms {
			vals := make(map[string]any, len(u.Values))
			for k, v := range u.Values {
				if s, ok := v.(string); ok {
					if n, isRef := refName(s); isRef {
						if iv, in := env[n]; in {
							v = iv
						}
					}
				}
				vals[k] = v
			}
			q.Uniforms[i] = Uniform{Name: u.Name, Values: vals}
		}
	}
	return q
}
