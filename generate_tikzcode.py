import json
import numpy as np

def draw_arrow(layer_type, next_type):
    if layer_type in ['Conv', 'ConvReLU'] and next_type in ['MaxPool', 'Conv', 'ConvReLU']:
        # 取消这两个层之间的箭头
        return False
    else:
        return True


def get_reference_shape(layers, reference_layer):
    for layer in layers:
        if (layer['type'] == 'Conv' or layer['type'] == 'ConvReLU') and reference_layer is None:
            # Set the first Conv layer as the reference
            reference_layer = layer
            input = reference_layer['inputs']
            return input[0]['shape']


def adjust_conv_shape(current_shape, reference_shape):
    min_size = 5
    max_size = 15
    height_ratio = current_shape[2] / reference_shape[2]
    depth_ratio = current_shape[3] / reference_shape[3]

    height_scale = min_size + (max_size - min_size) * height_ratio
    depth_scale = min_size + (max_size - min_size) * depth_ratio

    height = round(max(min_size, min(max_size, height_scale)))
    depth = round(max(min_size, min(max_size, depth_scale)))

    return height,depth


def generate_tikz_code(layers, config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)

    # TikZ 代码列表
    tikz_code = []

    # TikZ 头部和样式设置
    tikz_code.append(r"\documentclass[tikz,multi, border=10pt]{standalone}")
    tikz_code.append(r"\usetikzlibrary{quotes,arrows,positioning,3d, decorations.markings, calc}")
    tikz_code.append(r"\usepackage{xcolor}")
    tikz_code.append(r"\usepackage{../layer/Box}")
    tikz_code.append(r"\usepackage{../layer/Arrow}")
    tikz_code.append(r"\usepackage{../layer/Ball}")
    tikz_code.append(r"\begin{document}")
    tikz_code.append(r"\begin{tikzpicture}")
    tikz_code.append(r"\definecolor{lightred}{RGB}{167, 121, 121}")
    tikz_code.append(r"\definecolor{lightpink}{RGB}{228, 206, 206}")
    tikz_code.append(r"\definecolor{lightblue}{RGB}{166, 194, 212}")
    tikz_code.append(r"\definecolor{lightblue2}{RGB}{153, 164, 185}")
    tikz_code.append(r"\definecolor{lightgreen}{RGB}{188, 203, 178}")
    tikz_code.append(r"\definecolor{darkred}{RGB}{150, 80, 75}")
    tikz_code.append(r"\definecolor{lightgray}{RGB}{164, 152, 133}")

    # 用于conv层size调整的初始化
    reference_layer = None
    reference_shape = [1, 1, 60, 60]
    reference_shape = get_reference_shape(layers, reference_layer)

    # 初始化箭头参数（conv结合有关）
    arrow_check = False #false代表这一层不需要画箭头
    arrowidx = 0

    for layeridx in range(len(layers)):
        # layer type check
        layer = layers[layeridx]
        layer_type = layer['type']
        if layer != layers[-1]:
            next_type = layers[layeridx+1]['type']
        else:
            next_type = 'output'

        # box type define
        boxtype = 'BaseBox'
        if layer_type == 'ConvReLU':
            boxtype = 'ConvReLU'
        elif layer_type == 'Fc_combine':
            boxtype = 'PixelShuffle'


        # 获取当前layer信息
        inputs = layer['inputs']
        input_shape = inputs[0]['shape']
        input_shape = "*".join(map(str, input_shape))
        layer_config = config.get(layer_type, config.get('basebox', {}))

        color = layer_config.get('color', 'lightgreen')
        opacity = layer_config.get('opacity', 0.6)
        #xlabel = layer_config.get('xlabel', layer_type)
        zlabel = layer_config.get('zlabel', input_shape)
        scriptscale = layer_config.get('scriptscale', 1)
        border = layer_config.get('border', 'black')
        caption = layer_config.get('caption', layer_type)

        width = layer_config.get('width', 1)
        height = layer_config.get('height', 1)
        depth = layer_config.get('depth', 1)
        # 处理conv尺寸
        if (layer['type'] == 'ConvReLU' or layer['type'] == 'Conv' or layer['type'] == 'MaxPool') and layer != layers[0]:
            current_shape = inputs[0]['shape']
            #print(current_shape)
            height, depth = adjust_conv_shape(current_shape, reference_shape)

        # print(f"Drawing 3D shape for layer '{layer['name']}' of type '{layer_type}'")
        if layer == layers[0]:
            # input:sample image
            tikz_code.append(r"\node [canvas is zy plane at x=0] at (0,0,0) {\includegraphics[width=15cm]{sample.png}};")
            # first arrow
            tikz_code.append(
                rf"\pic[shift={{(1, 0, 0)}}] at (0,0,0) {{vecArrow={{name=input,caption=none, opacity=1,fill=black, scriptscale=0.2,length=3,tall=0.08,tipsize=0.2,border=none}}}};")
            # first layer
            tikz_code.append(
                rf"\pic[shift={{(5, 0, 0)}}] at (0, 0, 0) {{{boxtype}={{name={layeridx}, caption={caption},opacity={opacity}, zlabel={zlabel}, fill={color}, scriptscale={scriptscale}, width={width}, height={height}, depth={depth}, border={border}}}}};")
            # check if first layer needs arrow
            arrow_check = draw_arrow(layer_type, next_type)
            if arrow_check:
                tikz_code.append(
                    rf"\pic[shift={{(0, 0, 0)}}] at ({layeridx}-right) {{vecArrow={{name=arrow{arrowidx},caption=none, opacity=0.7,fill=black, scriptscale=0.2,length=3,tall=0.08,tipsize=0.2,border=none}}}};"
                )

        else:
            # last layer with arrow
            if arrow_check:
                if boxtype == 'PixelShuffle':
                    outputs = layer['outputs']
                    output_shape = outputs[0]['shape']
                    output_shape = "*".join(map(str, output_shape))
                    tikz_code.append(
                        rf"\pic[shift={{(.3,0,0)}}] at (arrow{arrowidx}-right) {{PixelShuffle={{ name={layeridx}, opacity=0.6,zlabel={input_shape}, caption=FC,fill=lightgreen, scriptscale=1, height=1, width=1,depth=12,border=black,dist=4,scalefactor=0.6,connectlineopacity=0.7, zzlabel={output_shape} ,ccaption=FC }}}};")
                else:
                    tikz_code.append(
                        rf"\pic[shift={{(.3, 0, 0)}}] at (arrow{arrowidx}-right) {{{boxtype}={{name={layeridx}, caption={caption}, opacity={opacity},  zlabel={zlabel}, fill={color}, scriptscale={scriptscale}, width={width}, height={height}, depth={depth}, border={border}}}}};"
                    )
                arrowidx += 1

            else:
                if boxtype == 'PixelShuffle':
                    outputs = layer['outputs']
                    output_shape = outputs[0]['shape']
                    output_shape = "*".join(map(str, output_shape))
                    tikz_code.append(
                        rf"\pic[shift={{(0,0,0)}}] at ({(layeridx-1)}-right) {{PixelShuffle={{ name={layeridx}, opacity=0.6,zlabel={input_shape}, caption=FC,fill=lightgreen, scriptscale=1, height=1, width=1,depth=12,border=black,dist=4,scalefactor=0.6,connectlineopacity=0.7, zzlabel={output_shape},ccaption=FC }}}};")
                else:
                    tikz_code.append(
                        rf"\pic[shift={{(0, 0, 0)}}] at ({layeridx-1}-right) {{{boxtype}={{name={layeridx}, caption={caption}, opacity={opacity},  zlabel={zlabel}, fill={color}, scriptscale={scriptscale}, width={width}, height={height}, depth={depth}, border={border}}}}};"
                    )
            arrow_check = draw_arrow(layer_type, next_type)
            if arrow_check:
                tikz_code.append(
                    rf"\pic[shift={{(0, 0, 0)}}] at ({layeridx}-right) {{vecArrow={{name=arrow{arrowidx},caption=none, opacity=0.7,fill=black, scriptscale=0.2,length=3,tall=0.08,tipsize=0.2,border=none}}}};"
                )



        if layer == layers[-1]:
            tikz_code.append(rf"\pic[shift={{(.5, 0, 0)}}] at (arrow{arrowidx}-right) {{Ball={{name=output,radius=3,fill=lightgray,opacity=0.6,logo = ,caption=output}}}};"
            )

    tikz_code.append("\end{tikzpicture}")
    tikz_code.append("\end{document}")
    return tikz_code