"""Fail closed on kernel launch metadata. No GPU imports."""
import re


def validate_abi(assembly, metadata):
    expected = dict(target=dict(backend='hip', arch='gfx906', warp_size=64),
                    num_warps=4, num_ctas=1, shared=0, global_scratch_size=0,
                    profile_scratch_size=0, launch_cooperative_grid=False,
                    enable_fp_fusion=False)
    for key, value in expected.items():
        if key not in metadata or metadata[key] != value:
            raise ValueError(f'Unsupported compiled metadata: {key}={metadata.get(key)!r}')
    blocks = re.findall(r'(?m)^\s*\.amdgpu_metadata\s*\n(.*?)^\s*\.end_amdgpu_metadata', assembly, re.S)
    if len(blocks) != 1:
        raise ValueError('Expected one AMDGPU metadata block')
    block = blocks[0]

    def scalar(key):
        values = re.findall(r'(?m)^\s*' + re.escape(key) + r':\s*([^\n]+)', block)
        if len(values) != 1:
            raise ValueError(f'Missing/ambiguous metadata: {key}')
        return values[0].strip().strip('"\'')

    for key, val in {'.kernarg_segment_size': '176', '.wavefront_size': '64',
                     '.max_flat_workgroup_size': '256', '.group_segment_fixed_size': '0',
                     '.name': metadata.get('name'),
                     'amdhsa.target': 'amdgcn-amd-amdhsa--gfx906'}.items():
        if scalar(key) != val:
            raise ValueError(f'Unsupported assembly metadata: {key}')
    args = block.split('  - .args:', 1)
    if len(args) != 2:
        raise ValueError('Missing kernel argument metadata')
    args = args[1].split('    .group_segment_fixed_size:', 1)[0]
    entries = re.split(r'(?m)^      - ', args)[1:]
    if len(entries) != 22:
        raise ValueError(f'Expected 20 dynamic + 2 scratch arguments, got {len(entries)}')
    for i, entry in enumerate(entries):
        fields = dict(re.findall(r'(\.[a-z_]+):\s*([^\s]+)', entry))
        want_kind = 'global_buffer' if i in (0, 5, 10, 15, 20, 21) else 'by_value'
        if (fields.get('.offset'), fields.get('.size'), fields.get('.value_kind')) != (str(i*8), '8', want_kind):
            raise ValueError(f'Kernel argument {i} is incompatible: {fields}')
    return {'argument_count': 22, 'kernarg_bytes': 176, 'block_threads': 256,
            'shared_bytes': 0, 'null_scratch_pointers': 2, 'scalar_type': 'int64'}
