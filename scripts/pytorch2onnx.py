import argparse
import torch
from ultralytics import YOLO
import os

def convert_pt_to_onnx(pt_path, yolo_version, onnx_path, use_fp16=False, use_dynamic=True):
    # Load the model
    model = YOLO(pt_path)
    
    # Set the export parameters
    export_params = {
        "format": "onnx",
        "opset": 12,
        "simplify": True,
    }

    # Handle FP16 and dynamic shape options
    if use_fp16:
        if not torch.cuda.is_available():
            raise RuntimeError("FP16 export requires GPU. Please run with GPU or use FP32.")
        
        # Force FP16 settings
        export_params.update({
            "half": True,
            "dynamic": False,  # FP16 is not compatible with dynamic shapes
            "device": 0  # Ensure GPU usage
        })
    else:
        export_params.update({
            "half": False,
            "dynamic": use_dynamic,
            "device": 'cpu' if not torch.cuda.is_available() else 0
        })

    # Print export configuration
    print("\nExport configuration:")
    print(f"- Precision: {'FP16' if export_params['half'] else 'FP32'}")
    print(f"- Dynamic shapes: {'Enabled' if export_params['dynamic'] else 'Disabled'}")
    print(f"- Device: {'GPU' if export_params['device'] == 0 else 'CPU'}")
    print(f"- Input path: {pt_path}")
    print(f"- Output path: {onnx_path}\n")

    # Export the model to ONNX
    try:
        model.export(**export_params)
        
        # Check if the file was created
        expected_path = os.path.splitext(pt_path)[0] + '.onnx'
        if os.path.exists(expected_path):
            # Rename the file if necessary
            if expected_path != onnx_path:
                os.rename(expected_path, onnx_path)
            
            # Verify the exported model size and characteristics
            model_size = os.path.getsize(onnx_path) / (1024 * 1024)  # Size in MB
            print(f"\nExport successful:")
            print(f"- Model saved at: {onnx_path}")
            print(f"- Model size: {model_size:.2f} MB")
            
            # Additional verification for FP16
            if use_fp16:
                print("- FP16 export completed (Note: Please verify with netron.app)")
        else:
            raise FileNotFoundError(f"Export completed, but file not found at: {expected_path}")
            
    except Exception as e:
        print(f"\nError during model conversion: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Convert YOLO PyTorch model to ONNX format")
    parser.add_argument("--pt_path", type=str, help="Path to the input .pt file")
    parser.add_argument("--yolo_version", type=str, 
                        choices=["v5n", "v5s", "v5m", "v5l", "v5x", 
                                 "v8n", "v8s", "v8m", "v8l", "v8x"],
                        help="YOLO model version and size (e.g., v8n for YOLOv8-nano)")
    parser.add_argument("--onnx_path", type=str, help="Path to save the output .onnx file")
    parser.add_argument("--fp16", action="store_true", help="Export in FP16 precision (requires GPU)")
    parser.add_argument("--no-dynamic", action="store_true", help="Disable dynamic shapes in export")
    
    args, unknown = parser.parse_known_args()

    # Handle positional arguments if named arguments are not provided
    if args.pt_path is None and len(unknown) >= 1:
        args.pt_path = unknown[0]
    if args.yolo_version is None and len(unknown) >= 2:
        args.yolo_version = unknown[1]
    if args.onnx_path is None and len(unknown) >= 3:
        args.onnx_path = unknown[2]

    # Validate required arguments
    if args.pt_path is None or args.yolo_version is None or args.onnx_path is None:
        parser.error("Missing required arguments. Please provide pt_path, yolo_version, and onnx_path.")

    convert_pt_to_onnx(args.pt_path, args.yolo_version, args.onnx_path, args.fp16, not args.no_dynamic)

if __name__ == "__main__":
    main()