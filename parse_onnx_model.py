import onnx
from onnx import shape_inference

def parse_onnx_model(onnx_file_path):
    print(f"Loading ONNX model from: {onnx_file_path}")

    # 加载模型并进行形状推断
    model = onnx.load(onnx_file_path)
    inferred_model = shape_inference.infer_shapes(model)
    graph = inferred_model.graph

    layers = []

    # 查找模型中所有输入输出张量的信息
    value_infos = {vi.name: vi for vi in graph.value_info}
    value_infos.update({vi.name: vi for vi in graph.input})
    value_infos.update({vi.name: vi for vi in graph.output})

    for node in graph.node:
        inputs = []
        outputs = []

        # 获取每个输入的形状
        for input_name in node.input:
            input_shape = None
            if input_name in value_infos:
                input_shape = [
                    dim.dim_value if dim.dim_value != 0 else 'unknown'
                    for dim in value_infos[input_name].type.tensor_type.shape.dim
                ]
            inputs.append({
                "name": input_name,
                "shape": input_shape
            })

        # 获取每个输出的形状
        for output_name in node.output:
            output_shape = None
            if output_name in value_infos:
                output_shape = [
                    dim.dim_value if dim.dim_value != 0 else 'unknown'
                    for dim in value_infos[output_name].type.tensor_type.shape.dim
                ]
            outputs.append({
                "name": output_name,
                "shape": output_shape
            })

        # 保存每个层的基本信息以及输入和输出的形状
        layer = {
            "name": node.name,
            "type": node.op_type,
            "inputs": inputs,
            "outputs": outputs,
        }
        #print(f"Parsed layer with shapes: {layer}")
        layers.append(layer)

    return layers