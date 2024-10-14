
def filter_layers(layers):
    print("Current layers in the model:")
    for idx, layer in enumerate(layers):
        print(f"{idx}: {layer['type']} ({layer['name']})")

    delete_indices = input("Enter the indices of layers you want to delete, separated by commas (or press Enter to keep all): ")
    if delete_indices:
        indices = list(map(int, delete_indices.split(',')))
        layers = [layer for idx, layer in enumerate(layers) if idx not in indices]
        #print(f"Layers after filtering: {[layer['name'] for layer in layers]}")
        print("Layer deleted successfully.")
    else:
        print("No layers deleted.")
    return layers


def process_layer_relationships(layers):
    print("Processing layer relationships...")
    new_layers = []
    skip_next = False

    for i in range(len(layers) - 1):
        if skip_next:
            skip_next = False
            continue

        current_layer = layers[i]
        next_layer = layers[i + 1]

        if current_layer['type'] == 'Conv' and next_layer['type'] == 'Relu':
            print(f"Combining Conv layer '{current_layer['name']}' with ReLU layer '{next_layer['name']}' into ConvReLU layer")
            current_layer['type'] = 'ConvReLU'
            new_layers.append(current_layer)
            skip_next = True
        else:
            new_layers.append(current_layer)

    if not skip_next and len(layers) > 0:
        new_layers.append(layers[-1])

    return new_layers


def combine_fc_layers(layers):
    print("Combining FullyConnected (Fc) layers...")
    new_layers = []
    last_fc_idx = None
    first_fc_idx = None

    # Find the last and first FullyConnected layer from the end
    for i in reversed(range(len(layers))):
        if layers[i]['type'] == 'FullyConnected' or layers[i]['type'] == 'Gemm':
            if last_fc_idx is None:
                last_fc_idx = i
            first_fc_idx = i
    print(last_fc_idx)
    if last_fc_idx is not None and first_fc_idx is not None:
        # Add layers before the first Fc layer
        new_layers.extend(layers[:first_fc_idx])
        # Combine the first and last Fc layers
        combined_fc_layer = layers[last_fc_idx].copy()
        combined_fc_layer['name'] = f"Fc_combine_{layers[first_fc_idx]['name']}_to_{layers[last_fc_idx]['name']}"
        combined_fc_layer['type'] = 'Fc_combine'
        combined_fc_layer['inputs'] = layers[first_fc_idx]['inputs']
        combined_fc_layer['outputs'] = layers[last_fc_idx]['inputs']
        new_layers.append(combined_fc_layer)
        print(f"Combined Fc layers from '{layers[first_fc_idx]['name']}' to '{layers[last_fc_idx]['name']}' into '{combined_fc_layer['name']}'")
        # Add layers after the last Fc layer
        new_layers.extend(layers[last_fc_idx + 1:])
    else:
        new_layers = layers

    return new_layers
