"""
ONNX Export Utility for NetSmith
Converts trained PyTorch models to ONNX format
"""

import sys
import json
import torch
from pathlib import Path

# Import the DynamicModel class from runner
from runner import DynamicModel


def export_to_onnx(run_path: str, output_path: str) -> dict:
    """
    Export a trained model to ONNX format

    Args:
        run_path: Path to the training run directory containing architecture.json and weights.pt
        output_path: Path where the ONNX file will be saved

    Returns:
        dict with status and any error messages
    """
    try:
        run_path = Path(run_path)
        output_path = Path(output_path)

        # Load architecture
        arch_file = run_path / 'architecture.json'
        if not arch_file.exists():
            return {
                'success': False,
                'error': f'Architecture file not found: {arch_file}'
            }

        with open(arch_file, 'r') as f:
            architecture = json.load(f)

        # Load weights
        weights_file = run_path / 'weights.pt'
        if not weights_file.exists():
            return {
                'success': False,
                'error': f'Weights file not found: {weights_file}\nPlease train the model first.'
            }

        # Get input shape from Input layer
        input_shape = None
        for node in architecture['nodes']:
            if node['data']['layerType'] == 'Input':
                input_shape = node['data']['params'].get('inputShape')
                break

        if not input_shape:
            return {
                'success': False,
                'error': 'Could not determine input shape from architecture'
            }

        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Build model
        model = DynamicModel(architecture)
        model.to(device)

        # Create dummy input with correct format
        # Convert HWC to NCHW format for PyTorch (batch_size=1, channels, height, width)
        if len(input_shape) == 3:
            # Assume HWC format from UI: [height, width, channels]
            dummy_input = torch.randn(1, input_shape[2], input_shape[0], input_shape[1]).to(device)
        elif len(input_shape) == 2:
            # 2D input: [height, width]
            dummy_input = torch.randn(1, input_shape[0], input_shape[1]).to(device)
        elif len(input_shape) == 1:
            # 1D input: [features]
            dummy_input = torch.randn(1, input_shape[0]).to(device)
        else:
            # Fallback
            dummy_input = torch.randn(1, 1, 28, 28).to(device)

        # Initialize lazy modules with dummy forward pass
        with torch.no_grad():
            try:
                _ = model(dummy_input)
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Failed to initialize model with dummy input: {str(e)}'
                }

        # Load trained weights after lazy modules are initialized
        state_dict = torch.load(weights_file, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        return {
            'success': True,
            'output_path': str(output_path),
            'input_shape': input_shape
        }

    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': f'{str(e)}\n\nTraceback:\n{traceback.format_exc()}'
        }


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: onnx_exporter.py <run_path> <output_path>")
        sys.exit(1)

    run_path = sys.argv[1]
    output_path = sys.argv[2]

    result = export_to_onnx(run_path, output_path)

    # Print result as JSON for parsing by extension
    print(json.dumps(result))

    if result['success']:
        sys.exit(0)
    else:
        sys.exit(1)
