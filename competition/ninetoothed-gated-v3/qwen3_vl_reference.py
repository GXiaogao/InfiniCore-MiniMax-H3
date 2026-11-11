"""CPU-only tiny Qwen3-VL diagnostic fixtures and independent mathematical reference.

No official weights, tokenizer, model implementation or device inference here.
Vision attention is frame-local, including in the separate video diagnostic.
"""
import math
import numpy as np

LAYERS = 2
DTYPES = ('float32', 'float16', 'bfloat16')
TOLERANCES = {'float32': (.0005, .0005), 'float16': (.02, .02), 'bfloat16': (.08, .08)}


def make_config(dtype):
    return {'model_type': 'qwen3_vl', 'architectures': ['Qwen3VLForConditionalGeneration'],
            'dtype': dtype, 'torch_dtype': dtype, 'tie_word_embeddings': False,
            'image_token_id': 60, 'video_token_id': 61,
            'vision_start_token_id': 62, 'vision_end_token_id': 63,
            'text_config': {'model_type': 'qwen3_vl_text', 'dtype': dtype, 'torch_dtype': dtype,
                            'vocab_size': 64, 'hidden_size': 64, 'intermediate_size': 128,
                            'num_hidden_layers': 3, 'num_attention_heads': 2,
                            'num_key_value_heads': 1, 'head_dim': 64, 'hidden_act': 'silu',
                            'attention_bias': False, 'attention_output_bias': False,
                            'mlp_bias': False, 'rms_norm_eps': 1e-6,
                            'max_position_embeddings': 128, 'rope_theta': 10000.,
                            'rope_scaling': {'rope_type': 'default', 'mrope_section': [16,8,8],
                                             'mrope_interleaved': True}},
            'vision_config': {'model_type': 'qwen3_vl', 'dtype': dtype, 'torch_dtype': dtype,
                              'hidden_size': 64, 'out_hidden_size': 64, 'depth': 2,
                              'num_heads': 1, 'intermediate_size': 128, 'in_channels': 3,
                              'patch_size': 2, 'temporal_patch_size': 2, 'spatial_merge_size': 2,
                              'num_position_embeddings': 9, 'deepstack_visual_indexes': [0,1],
                              'hidden_act': 'gelu_pytorch_tanh', 'layer_norm_eps': 1e-6}}


def make_weights(config, quantize):
    rng = np.random.default_rng(2026102403)
    weights = {}
    def put(name, value):
        weights[name] = quantize(np.asarray(value, np.float32))
    def linear(name, out_dim, in_dim, bias=True):
        put(name+'.weight', rng.normal(0, .35/np.sqrt(in_dim), (out_dim,in_dim)))
        if bias:
            put(name+'.bias', rng.normal(0, .03, out_dim))
    def norm(name, dim, bias):
        put(name+'.weight', 1+rng.normal(0, .06, dim))
        if bias:
            put(name+'.bias', rng.normal(0, .03, dim))
    v, t = config['vision_config'], config['text_config']
    vh, th, hd = v['hidden_size'], t['hidden_size'], t['head_dim']
    pd = v['in_channels']*v['temporal_patch_size']*v['patch_size']**2
    linear('model.visual.patch_embed.proj', vh, pd)
    weights['model.visual.patch_embed.proj.weight'] = weights['model.visual.patch_embed.proj.weight'].reshape(
        vh, v['in_channels'], v['temporal_patch_size'], v['patch_size'], v['patch_size'])
    put('model.visual.pos_embed.weight', rng.normal(0,.12,(v['num_position_embeddings'],vh)))
    for i in range(v['depth']):
        p = f'model.visual.blocks.{i}'
        for n in ('norm1','norm2'):
            norm(p+'.'+n, vh, True)
        linear(p+'.attn.qkv', 3*vh, vh)
        linear(p+'.attn.proj', vh, vh)
        linear(p+'.mlp.linear_fc1', v['intermediate_size'], vh)
        linear(p+'.mlp.linear_fc2', vh, v['intermediate_size'])
    merged = vh*v['spatial_merge_size']**2
    for i in range(-1,len(v['deepstack_visual_indexes'])):
        p = 'model.visual.'+('merger' if i == -1 else f'deepstack_merger_list.{i}')
        norm(p+'.norm', vh if i == -1 else merged, True)
        linear(p+'.linear_fc1', merged, merged)
        linear(p+'.linear_fc2', th, merged)
    put('model.language_model.embed_tokens.weight', rng.normal(0,.12,(t['vocab_size'],th)))
    for i in range(LAYERS):
        p = f'model.language_model.layers.{i}'
        for n in ('input_layernorm','post_attention_layernorm'):
            norm(p+'.'+n, th, False)
        for n, heads in [('q',t['num_attention_heads']),('k',t['num_key_value_heads']),('v',t['num_key_value_heads'])]:
            linear(p+'.self_attn.'+n+'_proj', heads*hd, th, False)
        for n in ('q','k'):
            norm(p+'.self_attn.'+n+'_norm', hd, False)
        linear(p+'.self_attn.o_proj', th, t['num_attention_heads']*hd, False)
        for n in ('gate','up'):
            linear(p+'.mlp.'+n+'_proj', t['intermediate_size'], th, False)
        linear(p+'.mlp.down_proj', th, t['intermediate_size'], False)
    return weights


def patch_positions(grid, merge):
    nt, h, w = grid
    if h % merge or w % merge:
        raise ValueError('Grid must be divisible by merge size')
    rows, cols = np.indices((h,w))
    def order(a):
        return np.tile(a.reshape(h//merge,merge,w//merge,merge).transpose(0,2,1,3).ravel(), nt)
    return order(rows), order(cols)


def interpolate_positions(table, grid, merge):
    _, h, w = grid
    ys = np.linspace(0, table.shape[0]-1, h, dtype=np.float32)
    xs = np.linspace(0, table.shape[1]-1, w, dtype=np.float32)
    y0, x0 = ys.astype(int), xs.astype(int)
    y1, x1 = np.minimum(y0+1,table.shape[0]-1), np.minimum(x0+1,table.shape[1]-1)
    fy, fx = (ys-y0).astype(np.float32)[:,None,None], (xs-x0).astype(np.float32)[None,:,None]
    resized = ((1-fy)*((1-fx)*table[y0[:,None],x0]+fx*table[y0[:,None],x1])
               + fy*((1-fx)*table[y1[:,None],x0]+fx*table[y1[:,None],x1]))
    rows, cols = patch_positions(grid,merge)
    return resized[rows,cols].astype(np.float32)


def make_case(config, index, quantize, video=False):
    rng = np.random.default_rng(2026102404+index+100*video)
    # Each structured segment independently specifies its expected merged token positions.
    grids = [(2,2,4)] if video else ([ (1,4,6) ] if index == 1 else ([(1,2,4),(1,4,2)] if index == 2 else []))
    segments = ([('text',2),('visual',0),('text',1),('visual',0),('text',2)] if video else
                ([('text',2),('visual',0),('text',3)] if index == 1 else
                 ([('text',1),('visual',0),('text',2),('visual',1),('text',3)] if index == 2 else
                  [('text',7 if index == 0 else 5)])))
    v = config['vision_config']
    merge = v['spatial_merge_size']
    media = [{'grid':tuple(g),'bounds':[], 'kind':2 if video else 1,
              'pixels':quantize(rng.normal(0,.5,(int(np.prod(g)),v['in_channels']*v['temporal_patch_size']*v['patch_size']**2)).astype(np.float32))}
             for g in grids]
    positions, types = [], []
    current, offset = 0, 0
    for kind, number in segments:
        if kind == 'text':
            p = np.tile(np.arange(current,current+number), (3,1))
            current += number
            types.extend([0]*number)
        else:
            _, h, w = grids[number]
            ph, pw = h//merge, w//merge
            rows, cols = np.indices((ph,pw))
            p = np.stack([np.full(ph*pw,current),rows.ravel()+current,cols.ravel()+current])
            current += max(ph,pw)
            types.extend([2 if video else 1]*(ph*pw))
            media[number]['bounds'].append((offset,offset+ph*pw))
        positions.append(p)
        offset += p.shape[1]
    ids = rng.integers(0,60,(1,offset),dtype=np.int64)
    types = np.asarray([types],np.int64)
    ids[types==1], ids[types==2] = 60, 61
    return {'name':('video_two_frames' if video else ('text','image_rectangle','two_images','text_after_media')[index])+f'_{index}',
            'input_ids':ids,'mm_token_type_ids':types,'positions':np.concatenate(positions,axis=1).astype(np.int64),
            'media':media}


def attention(query, key, value, causal=False):
    groups = query.shape[1]//key.shape[1]
    k, v = np.repeat(key,groups,axis=1), np.repeat(value,groups,axis=1)
    score = np.einsum('qhd,khd->hqk',query,k)*np.float32(query.shape[-1]**-.5)
    if causal:
        score = np.where(np.triu(np.ones(score.shape[-2:],bool),1),-np.inf,score)
    prob = np.exp(score-score.max(axis=-1,keepdims=True))
    prob /= prob.sum(axis=-1,keepdims=True)
    return np.einsum('hqk,khd->qhd',prob,v).astype(np.float32)


def rotate(x, angles, quantize):
    cos, sin = quantize(np.cos(angles)), quantize(np.sin(angles))
    left,right = np.split(x,2,axis=-1)
    return quantize(np.concatenate((left*cos[:,None]-right*sin[:,None],right*cos[:,None]+left*sin[:,None]),axis=-1))


def text_rotary(x, positions, sections, theta, quantize):
    half = x.shape[-1]//2
    axes = np.zeros(half,np.int64)
    axes[1:3*sections[1]:3] = 1
    axes[2:3*sections[2]:3] = 2
    frequency = np.power(np.float32(theta),-np.arange(half,dtype=np.float32)/np.float32(half))
    angles = positions[axes].T.astype(np.float32)*frequency
    return rotate(x,angles,quantize)


def _math(weights, quantize):
    def linear(name,x):
        w = weights[name+'.weight']
        y = x@w.reshape(w.shape[0],-1).T
        if name+'.bias' in weights:
            y += weights[name+'.bias']
        return quantize(y)
    def norm(name,x,rms=False):
        centered = x if rms else x-x.mean(-1,keepdims=True)
        y = centered/np.sqrt(np.mean(centered*centered,axis=-1,keepdims=True)+np.float32(1e-6))
        y = y*weights[name+'.weight']
        if name+'.bias' in weights:
            y += weights[name+'.bias']
        return quantize(y)
    return linear,norm


def gelu(x,tanh=False):
    if tanh:
        return .5*x*(1+np.tanh(np.float32(np.sqrt(2/np.pi))*(x+np.float32(.044715)*x**3)))
    erf = np.fromiter((math.erf(float(a)) for a in (x/np.float32(np.sqrt(2))).ravel()), np.float32, x.size).reshape(x.shape)
    return .5*x*(1+erf)


def vision_reference(config,weights,media,quantize):
    v = config['vision_config']; q = quantize
    linear,norm = _math(weights,q)
    h, heads = v['hidden_size'],v['num_heads']
    d, merge = h//heads,v['spatial_merge_size']
    grid = media['grid']; nt, gh, gw = grid
    table = weights['model.visual.pos_embed.weight'].reshape(3,3,h)
    x = q(linear('model.visual.patch_embed.proj',media['pixels'])+q(interpolate_positions(table,grid,merge)))
    rows,cols = patch_positions(grid,merge)
    freq = np.power(np.float32(10000),-np.arange(d//4,dtype=np.float32)/np.float32(d//4))
    angles = np.concatenate((rows[:,None]*freq,cols[:,None]*freq),axis=1).astype(np.float32)
    def merger(name,hidden,post):
        y = norm(name+'.norm', hidden.reshape(-1,h*merge**2) if post else hidden).reshape(-1,h*merge**2)
        return linear(name+'.linear_fc2',q(gelu(linear(name+'.linear_fc1',y))))
    deep=[]
    for layer in range(v['depth']):
        p=f'model.visual.blocks.{layer}'
        qkv=linear(p+'.attn.qkv',norm(p+'.norm1',x)).reshape(-1,3,heads,d)
        query,key=rotate(qkv[:,0],angles,q),rotate(qkv[:,1],angles,q)
        # Vision attention must never cross frame boundaries.
        out=np.concatenate([attention(query[i*gh*gw:(i+1)*gh*gw],key[i*gh*gw:(i+1)*gh*gw],qkv[i*gh*gw:(i+1)*gh*gw,2]) for i in range(nt)])
        x=q(x+linear(p+'.attn.proj',q(out).reshape(-1,h)))
        y=linear(p+'.mlp.linear_fc1',norm(p+'.norm2',x))
        x=q(x+linear(p+'.mlp.linear_fc2',q(gelu(y,tanh=True))))
        if layer in v['deepstack_visual_indexes']:
            idx=v['deepstack_visual_indexes'].index(layer)
            deep.append(merger(f'model.visual.deepstack_merger_list.{idx}',x,True))
    return merger('model.visual.merger',x,False),deep


def encoder_reference(config,weights,case,quantize):
    q=quantize; t=config['text_config']; linear,norm=_math(weights,q)
    x=weights['model.language_model.embed_tokens.weight'][case['input_ids'][0]].copy()
    deep=[]
    for media in case['media']:
        pooled,features=vision_reference(config,weights,media,q)
        offset=0
        for start,end in media['bounds']:
            n=end-start
            x[start:end]=pooled[offset:offset+n]
            deep.append((start,end,[f[offset:offset+n] for f in features]))
            offset+=n
    for layer in range(LAYERS):
        p=f'model.language_model.layers.{layer}'
        y=norm(p+'.input_layernorm',x,True)
        s,d=x.shape[0],t['head_dim']
        query=linear(p+'.self_attn.q_proj',y).reshape(s,t['num_attention_heads'],d)
        key=linear(p+'.self_attn.k_proj',y).reshape(s,t['num_key_value_heads'],d)
        value=linear(p+'.self_attn.v_proj',y).reshape(s,t['num_key_value_heads'],d)
        query=norm(p+'.self_attn.q_norm',query,True); key=norm(p+'.self_attn.k_norm',key,True)
        query=text_rotary(query,case['positions'],t['rope_scaling']['mrope_section'],t['rope_theta'],q)
        key=text_rotary(key,case['positions'],t['rope_scaling']['mrope_section'],t['rope_theta'],q)
        x=q(x+linear(p+'.self_attn.o_proj',q(attention(query,key,value,causal=True)).reshape(s,-1)))
        y=norm(p+'.post_attention_layernorm',x,True)
        gate,up=linear(p+'.mlp.gate_proj',y),linear(p+'.mlp.up_proj',y)
        x=q(x+linear(p+'.mlp.down_proj',q(up*gate/(1+np.exp(-gate)))))
        for start,end,features in deep:
            if layer<len(features):
                x[start:end]=q(x[start:end]+features[layer])
    return x[None]
