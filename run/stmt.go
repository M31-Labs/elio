package run

import (
	"fmt"

	"m31labs.dev/elio/ir"
)

func (ev *evaluator) execBlock(body []ir.Stmt) (flow, error) {
	for _, s := range body {
		f, err := ev.execStmt(s)
		if err != nil {
			return flowNormal, err
		}
		if f != flowNormal {
			return f, nil
		}
	}
	return flowNormal, nil
}

func (ev *evaluator) execStmt(s ir.Stmt) (flow, error) {
	switch x := s.(type) {
	case ir.Let:
		v, err := ev.eval(x.Value)
		if err != nil {
			return flowNormal, err
		}
		ev.locals[x.Name] = v
	case ir.Var:
		v, err := ev.eval(x.Init)
		if err != nil {
			return flowNormal, err
		}
		ev.locals[x.Name] = v
	case ir.Assign:
		v, err := ev.eval(x.Value)
		if err != nil {
			return flowNormal, err
		}
		if x.Op != "" {
			cur, err := ev.eval(x.Target)
			if err != nil {
				return flowNormal, err
			}
			if v, err = binop(x.Op, cur, v); err != nil {
				return flowNormal, err
			}
		}
		if err := ev.assign(x.Target, v); err != nil {
			return flowNormal, err
		}
	case ir.Return:
		return flowReturn, nil
	case ir.Break:
		return flowBreak, nil
	case ir.Barrier:
		if ev.barrier == nil {
			return flowNormal, fmt.Errorf("run: barrier outside a workgroup (kernel declares no shared memory)")
		}
		ev.barrier.wait()
	case ir.Do:
		_, err := ev.eval(x.Expr)
		return flowNormal, err
	case ir.If:
		c, err := ev.eval(x.Cond)
		if err != nil {
			return flowNormal, err
		}
		if toBool(c) {
			return ev.execBlock(x.Then)
		}
		if len(x.Else) > 0 {
			return ev.execBlock(x.Else)
		}
	case ir.For:
		if x.Init != nil {
			if _, err := ev.execStmt(x.Init); err != nil {
				return flowNormal, err
			}
		}
		for {
			c, err := ev.eval(x.Cond)
			if err != nil {
				return flowNormal, err
			}
			if !toBool(c) {
				break
			}
			f, err := ev.execBlock(x.Body)
			if err != nil {
				return flowNormal, err
			}
			if f == flowReturn {
				return flowReturn, nil
			}
			if f == flowBreak {
				break
			}
			if x.Post != nil {
				if _, err := ev.execStmt(x.Post); err != nil {
					return flowNormal, err
				}
			}
		}
	case ir.While:
		for {
			c, err := ev.eval(x.Cond)
			if err != nil {
				return flowNormal, err
			}
			if !toBool(c) {
				break
			}
			f, err := ev.execBlock(x.Body)
			if err != nil {
				return flowNormal, err
			}
			if f == flowReturn {
				return flowReturn, nil
			}
			if f == flowBreak {
				break
			}
		}
	default:
		return flowNormal, fmt.Errorf("run: unsupported statement %T", s)
	}
	return flowNormal, nil
}

func (ev *evaluator) assign(target ir.Expr, v any) error {
	switch t := target.(type) {
	case ir.Name:
		ev.locals[t.Name] = v
		return nil
	case ir.Index:
		base, err := ev.eval(t.E)
		if err != nil {
			return err
		}
		idx, err := ev.eval(t.Idx)
		if err != nil {
			return err
		}
		i := int(toFloat(idx))
		switch arr := base.(type) {
		case []any:
			if i < 0 || i >= len(arr) {
				return fmt.Errorf("run: index %d out of range", i)
			}
			arr[i] = v
		case []float64:
			if i < 0 || i >= len(arr) {
				return fmt.Errorf("run: index %d out of range", i)
			}
			arr[i] = toFloat(v)
		default:
			return fmt.Errorf("run: cannot index-assign into %T", base)
		}
		return nil
	case ir.Member:
		base, err := ev.eval(t.E)
		if err != nil {
			return err
		}
		m, ok := base.(map[string]any)
		if !ok {
			return fmt.Errorf("run: cannot assign to .%s of %T", t.Field, base)
		}
		m[t.Field] = v
		return nil
	}
	return fmt.Errorf("run: unsupported assign target %T", target)
}
