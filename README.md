
# **ONNX to TikZ Converter User Guide**

## **Overview**

This project provides a tool to convert ONNX neural network models into a TikZ diagram, which can be used for visualizing the structure of the network in LaTeX documents. The tool parses ONNX models, allows layer filtering, combines specific types of layers (e.g., Conv + ReLU), and generates a TikZ representation of the network.

---

## **Installation**

### **Requirements:**
- Python 3.x
- ONNX
- NumPy
- ONNX Runtime (optional, for more advanced ONNX functionality)

You can install the required Python libraries using `pip`:

```bash
pip install onnx numpy onnxruntime
```

### **Files Included:**
1. **Main.py** – The entry point script that coordinates the entire process.
2. **parse_onnx_model.py** – Contains logic to parse ONNX models and extract layer information.
3. **filter_layers.py** – Handles the logic for filtering and combining layers based on user input.
4. **generate_tikzcode.py** – Generates TikZ code for visualizing the ONNX model.
5. **layer_shapes.json** – Contains configuration for the shapes, colors, and sizes of different layer types in the TikZ output.

---

## **Usage**

### **Running the Script**

Run the main script `Main.py` to start the ONNX to TikZ conversion:

```bash
python Main.py
```

### **Steps**

1. **Input the ONNX File**:  
   You will be prompted to provide the path to your ONNX file.
   
   Example:
   ```
   Enter the path to the ONNX file: my_model.onnx
   ```

2. **Filter Layers**:  
   The script will display a list of all layers present in the ONNX model and prompt you to choose any layers you want to delete. If you don’t want to remove any layers, just press Enter.

   Example:
   ```
   Current layers in the model:
   0: Conv (conv1)
   1: Relu (relu1)
   2: MaxPool (maxpool1)
   ...
   
   Enter the indices of layers you want to delete, separated by commas (or press Enter to keep all): 1
   ```

3. **Combining Layers**:
   - **Conv + ReLU Combination**: You will be asked if you want to combine Conv and ReLU layers into a single "ConvReLU" layer.
   - **Fully Connected Layers Combination**: You can combine the final fully connected layers (`Gemm` or `FullyConnected` layers).

   Example:
   ```
   Do you want to combine Conv and ReLU layers? (yes/no): yes
   Do you want to combine all fc layers? (caution: Only applicable to the combination of the fc layer at the end of the neural network): yes
   ```

4. **Generate TikZ Code**:  
   After filtering and combining layers, the TikZ code for the neural network diagram will be generated and saved to a file named `network_diagram.tex` in the `Tikz_output/` directory.

---

## **File Descriptions**

### **Main.py**
Coordinates the conversion process:
1. Loads the ONNX model.
2. Prompts the user for filtering and combining layers.
3. Generates TikZ code based on the processed layer information.
   
### **parse_onnx_model.py**
This file handles the parsing of the ONNX model:
- It extracts the layers, including their input and output shapes, and formats this information for further processing.

### **filter_layers.py**
Includes two main functionalities:
1. **filter_layers**: Lets the user interactively select which layers to delete from the model.
2. **process_layer_relationships** and **combine_fc_layers**: Handle the combination of Conv + ReLU and Fully Connected layers, respectively.

### **generate_tikzcode.py**
Generates the LaTeX TikZ code to represent the network layers, utilizing the layout, colors, and sizes defined in the `layer_shapes.json` file. This code defines the structure and style of the network diagram.

### **layer_shapes.json**
Stores the configuration for different layer types, including properties like width, height, depth, and color. These configurations are applied during the TikZ code generation.

---

## **Output**

The output is a LaTeX file (`network_diagram.tex`) containing TikZ code to visualize the ONNX model structure. You can include this file in your LaTeX documents to produce the network diagram.

### **Example of including TikZ in LaTeX**:

```latex
\documentclass{article}
\usepackage{tikz}
\begin{document}
    \input{Tikz_output/network_diagram.tex}
\end{document}
```

---

## **Customization**

- You can modify the `layer_shapes.json` file to customize the colors, sizes, and styles of different layers in the generated TikZ code.
- The TikZ code generation logic can also be adjusted in the `generate_tikzcode.py` file if needed.

---
