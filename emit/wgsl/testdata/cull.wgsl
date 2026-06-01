struct CullUniforms {
  planes : array<vec4<f32>, 6>,
  vertexCount : u32,
  radius : f32,
  _pad0 : vec2<f32>,
};

struct InstanceRecord {
  model : mat4x4<f32>,
  pickData : vec4<u32>,
};

@group(0) @binding(0) var<uniform> cull : CullUniforms;
@group(0) @binding(1) var<storage, read> input : array<InstanceRecord>;
@group(0) @binding(2) var<storage, read_write> output : array<InstanceRecord>;
@group(0) @binding(3) var<storage, read_write> drawArgs : array<atomic<u32>, 4>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if ((i >= arrayLength(&input))) {
    return;
  }
  let record = input[i];
  let center = record.model[3].xyz;
  var inside = true;
  for (var p : i32 = 0; (p < 6); p = (p + 1)) {
    let plane = cull.planes[p];
    if (((dot(plane.xyz, center) + plane.w) < -cull.radius)) {
      inside = false;
      break;
    }
  }
  if (inside) {
    let slot = atomicAdd(&drawArgs[1], 1u);
    output[slot] = record;
  }
}
