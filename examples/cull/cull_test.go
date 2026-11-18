package cull

import (
	"testing"

	"m31labs.dev/gosx/render/compute"
	"m31labs.dev/gosx/render/gpu"
	"m31labs.dev/gosx/render/gpu/headless"
)

func TestPooledCullBindsOnlyLiveInstances(t *testing.T) {
	dev, _ := headless.New(16, 16)
	pass, err := New(dev, "culled")
	if err != nil {
		t.Fatal(err)
	}
	data := make([]byte, 32*compute.InstanceRecordStride)
	var firstInput gpu.Buffer
	var previousGroup gpu.BindGroup
	var previousCount int
	for _, count := range []int{8, 2, 2, 7} {
		pass.SetFrame(data, count, [6][4]float32{}, 3, 0)
		encoder := dev.CreateCommandEncoder()
		published := map[string]compute.GPUResource{}
		if err := pass.Record(compute.PassContext{Device: dev, Encoder: encoder, Publish: func(r compute.GPUResource) { published[r.Name] = r }}); err != nil {
			t.Fatal(err)
		}
		dev.Queue().Submit(encoder.Finish())
		entries := pass.res.bindGroup.(*headless.BindGroup).Desc().Entries
		for _, e := range entries {
			if (e.Binding == 1 || e.Binding == 2) && e.Size != count*compute.InstanceRecordStride {
				t.Fatalf("count %d: binding %d exposes %d bytes", count, e.Binding, e.Size)
			}
		}
		if firstInput != nil && pass.res.inputBuf != firstInput {
			t.Fatal("reallocated pooled input within capacity")
		}
		if got := pass.res.inputBuf.(*headless.Buffer).LastWriteSize(); got != count*compute.InstanceRecordStride {
			t.Fatalf("count %d: uploaded %d bytes", count, got)
		}
		if got := published["culled.instances"].Count; got != count {
			t.Fatalf("published %d instances, want %d", got, count)
		}
		if previousCount == count && previousGroup != pass.res.bindGroup {
			t.Fatal("rebuilt bind group for the same live count")
		}
		firstInput, previousGroup, previousCount = pass.res.inputBuf, pass.res.bindGroup, count
	}
	pass.SetFrame(data[:compute.InstanceRecordStride], 2, [6][4]float32{}, 3, 0)
	if err := pass.Record(compute.PassContext{}); err == nil {
		t.Fatal("accepted a count beyond the supplied records")
	}
}
