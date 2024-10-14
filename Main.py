import onnx
from onnx import numpy_helper
import json
import onnxruntime as ort
import numpy as np
from parse_onnx_model import parse_onnx_model
from filter_layers import filter_layers, process_layer_relationships, combine_fc_layers
from generate_tikzcode import generate_tikz_code

# Step 1: Parse the ONNX file and extract network structure
# Step 2: Interact with the user to filter layers
# Step 3: Generate TikZ code with automatic drawing

# Main function to coordinate the process
def main():
    onnx_file_path = input("Enter the path to the ONNX file: ")
    config_file = "layer_shapes.json"
    print("Loading the layer configuration JSON file...\n")

    print("Parsing ONNX model...")
    layers = parse_onnx_model(onnx_file_path)
    print("Filtering layers...")
    layers = filter_layers(layers)

    process_relationships = input(
        "Do you want to combine Conv and ReLU layers? (yes/no): ").strip().lower()
    if process_relationships == 'yes':
        print("Processing layer relationships...")
        layers = process_layer_relationships(layers)

    process_relationships = input(
        "Do you want to combine all fc layers?(\033[91mcaution:\033[0m Only applicable to the combination of the fc layer at the end of the neural network. If it is not applicable, please enter no) (yes/no): ").strip().lower()
    if process_relationships == 'yes':
        print("Processing fc combination...")
        layers = combine_fc_layers(layers)

    print("Generating TikZ code...")
    tikz_code = generate_tikz_code(layers, config_file)

    with open('Tikz_output/network_diagram.tex', 'w') as tex_file:
        tex_file.write("\n".join(tikz_code))

    print("TikZ code has been generated and saved to 'network_diagram.tex'")


if __name__ == "__main__":
    main()