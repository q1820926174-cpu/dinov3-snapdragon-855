"""Representative real DINOv3 operators; no synthetic peak-throughput claims."""
import copy,json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,shape_inference,utils
import onnxruntime as ort

root=Path('gpu_operator_probes');root.mkdir(exist_ok=True)
m=shape_inference.infer_shapes(onnx.load('dinov3_gpu_adaptive.onnx'))
onnx.save(m,root/'inferred.onnx')
shapes={v.name:[d.dim_value for d in v.type.tensor_type.shape.dim] for v in list(m.graph.value_info)+list(m.graph.input)+list(m.graph.output)}
inits={w.name:w for w in m.graph.initializer};nodes=list(m.graph.node)
cases=[]
for label,pattern in [('expand56','/stages.0/stages.0.2/expand/Conv'),('expand14','/stages.2/stages.2.2/expand/Conv'),
                      ('expand7','/stages.3/stages.3.4/expand/Conv'),('depthwise56','/stages.0/stages.0.2/depthwise/Conv')]:
    n=next(n for n in nodes if n.name==pattern)
    w=next(w for w in m.graph.initializer if w.name==n.input[1]);dims=list(w.dims)
    output=shapes[n.output[0]];mac=int(np.prod(output[2:])*np.prod(dims))
    cases.append({'name':label,'input_name':n.input[0],'output_name':n.output[0],'mac':mac})
cases.append({'name':'norm56','input_name':next(n for n in nodes if n.name=='/stages.0/stages.0.2/depthwise/Conv').output[0],
              'output_name':next(n for n in nodes if n.name=='/stages.0/stages.0.2/norm/Add_1').output[0],'mac':0})
probe=copy.deepcopy(m)
needed=list(dict.fromkeys([c['input_name'] for c in cases]+[c['output_name'] for c in cases]))
del probe.graph.output[:]
for name in needed:probe.graph.output.append(helper.make_tensor_value_info(name,onnx.TensorProto.FLOAT,shapes[name]))
opts=ort.SessionOptions();opts.intra_op_num_threads=4
s=ort.InferenceSession(probe.SerializeToString(),opts,providers=['CPUExecutionProvider'])
r=json.loads(Path('six_fp32/feature_manifest.jsonl').read_text(encoding='utf-8').splitlines()[0])
x=np.fromfile(Path('six_game_set')/r['raw_file'],np.float32).reshape(r['shape']).transpose(0,3,1,2)
values=dict(zip(needed,s.run(needed,{'pixel_values':x})))
lines=[]
for c in cases:
    p=root/c['name'];p.mkdir(exist_ok=True)
    utils.extract_model(str(root/'inferred.onnx'),str(p/'model.onnx'),[c['input_name']],[c['output_name']])
    single=onnx.load(p/'model.onnx')
    for n in single.graph.node:
        for i,name in enumerate(n.input):
            if name==c['input_name']:n.input[i]='input'
        for i,name in enumerate(n.output):
            if name==c['output_name']:n.output[i]='output'
    single.graph.input[0].name='input';single.graph.output[0].name='output'
    onnx.checker.check_model(single);onnx.save(single,p/'model.onnx')
    np.ascontiguousarray(values[c['input_name']]).tofile(p/'input.raw')
    np.ascontiguousarray(values[c['output_name']]).tofile(p/'reference.raw')
    c['input_shape']=shapes[c['input_name']];c['output_shape']=shapes[c['output_name']]
    _,ci,hi,wi=c['input_shape'];_,co,ho,wo=c['output_shape']
    lines.append(f"{c['name']} {ci} {hi} {wi} {co} {ho} {wo} {c['mac']}")
(root/'cases.txt').write_text('\n'.join(lines)+'\n')
(root/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
print('Exported real operator probes:',len(cases))
