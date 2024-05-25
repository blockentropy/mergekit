import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import bitsandbytes as bnb
import click
import torch
from peft.tuners.lora import QuantLinear
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel

from mergekit.card import generate_card_lora
from mergekit.common import ModelReference
from mergekit.io import LazyTensorLoader


def find_all_linear_names(model: PreTrainedModel) -> List[str]:
    cls = (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt, torch.nn.Linear, QuantLinear)

    names = []
    for name, module in model.named_modules():
        if (
            isinstance(module, cls)
            or "Linear" in module.__class__.__name__
            and module.__class__.__name__ not in ("LlamaLinearScalingRotaryEmbedding",)
        ):
            names.append(name)

    return names


def get_linear_module_names(model_id: str) -> List[str]:
    model = AutoModelForCausalLM.from_pretrained(
        model_id, state_dict={}, device_map="meta"
    )  # avoid loading weights as we won't need them
    linear_module_names = find_all_linear_names(model)

    return linear_module_names


def reconstruct_invocation(args):
    """
    Reconstructs the command-line invocation string based on the given arguments stored in a dictionary.

    Parameters:
    - args: A dictionary containing the command arguments with keys matching the parameter names.
      Expected keys are 'base_model' 'out_path', 'no_lazy_unpickle', 'model_name' and 'device'.

    Returns:
    - The reconstructed command-line invocation string.
    """
    # Provide a default value for out_path if it's not in the dictionary
    out_path = args.get("out_path", "OUTPUT_PATH")

    invocation = f"mergekit-reduce-layers {args['base_model']} {out_path}"
    if args.get("no_lazy_unpickle"):
        invocation += " --no-lazy-unpickle"
    if args.get("model_name"):
        invocation += f" --model_name={args['model_name']}"
    if args.get("device"):
        invocation += f" --device={args['device']}"

    return invocation


@click.command("mergekit-reduce-layers")
@click.argument("base_model", type=str)
@click.argument("out_path", type=click.Path())
@click.option(
    "--no-lazy-unpickle",
    is_flag=True,
    help="Disable lazy unpickler (more stable, higher memory usage)",
)
@click.option(
    "--model_name",
    type=str,
    default=None,
    help="Name of the resulting model (shown in the model card)",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="PyTorch device to perform SVD computation on",
)
def main(
    base_model: str,
    out_path: str,
    no_lazy_unpickle: bool,
    model_name: str,
    device: str,
) -> None:
    """
    Computes and sorts the removal of layers that can be repeated while minimizing loss, saving a merge plan specified output path.

    \b
    Arguments:
    BASE_MODEL - the model ID or path to use as the base model.
    OUT_PATH - the output path where the merge plan will be saved.
    """

    invocation_args = {
        "base_model": base_model,
        "device": device,
        "out_path": out_path,
        "model_name": model_name,
        "no_lazy_unpickle": no_lazy_unpickle,
    }

    os.makedirs(out_path, exist_ok=True)

    base_model_ref = ModelReference.parse(base_model)

    linear_module_names = get_linear_module_names(base_model_ref.model.path)

    base_loader = LazyTensorLoader(
        base_model_ref.tensor_index(), lazy_unpickle=(not no_lazy_unpickle)
    )

    layer_weights = {}
    layer_norm_sums = {}
    print(f"Loading weights for {len(linear_module_names)} modules.")
    for layer_name in linear_module_names:
        # Ensure the key used here matches the one used in the comparison loop
        try:
            layer_weights[layer_name] = base_loader.get_tensor(layer_name + ".weight")
        except Exception as e:
            print(f"Failed to load weight for {layer_name}: {e}")
    print("Weights loaded.")

    layer_indices = sorted(set(int(name.split('.')[2]) for name in linear_module_names if name.startswith('model.layers')))

    print(layer_indices)
    # Loop to compare equivalent parts of adjacent layers

    # Loop to compare equivalent parts of adjacent layers
    for i in layer_indices:
        if i + 1 in layer_indices:
            layer_base = f"model.layers.{i}"
            next_layer_base = f"model.layers.{i + 1}"
            sum_norm = 0  # Initialize the sum for this layer transition

            # Fetch all components for each layer
            current_parts = [name for name in linear_module_names if name.startswith(layer_base)]
            next_parts = [name for name in linear_module_names if name.startswith(next_layer_base)]

            # Match components based on the sub-component name only
            for part in current_parts:
                # Correctly extracting the sub-component name after the layer number
                part_suffix = '.'.join(part.split('.')[3:])  # Fixes the typo here

                # Build the corresponding part name in the next layer
                corresponding_part = f"{next_layer_base}.{part_suffix}"

                if corresponding_part in next_parts:
                    current_weight = layer_weights.get(part)
                    next_weight = layer_weights.get(corresponding_part)

                    if current_weight is not None and next_weight is not None:
                        # Compute delta weights
                        delta_weight = next_weight - current_weight

                        # Calculate the Frobenius norm of the delta weights
                        fro_norm = torch.norm(delta_weight, 'fro')
                        layer_weights[(part, corresponding_part)] = fro_norm
                        sum_norm += fro_norm  # Add this norm to the sum for the current layer transition
                        print(f'Frobenius norm between {part} and {corresponding_part}: {fro_norm}')
                    else:
                        print(f"Missing weights for comparison: {part} or {corresponding_part}")
                else:
                    print(f"No corresponding part found in next layer for {part}")

            layer_norm_sums[f"{layer_base} to {next_layer_base}"] = sum_norm

    print("Layer to Layer Transition Norms Summary:")
    for key, value in layer_norm_sums.items():
        print(f"{key}: Total Frobenius Norm = {value}")

    #with open(os.path.join(out_path, "mergestuff.json"), "w") as f:
    #    json.dump(lora_config, f, indent=2)

    #save_file(lora_weights, os.path.join(out_path, "adapter_model.safetensors"))

    invocation_args.pop("out_path")  # don't include out_path for privacy
    invocation = reconstruct_invocation(invocation_args)

    logging.info(f"Model plan saved to {out_path}")


if __name__ == "__main__":
    main()
