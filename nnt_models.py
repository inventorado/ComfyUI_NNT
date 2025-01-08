# NNT_models.py
import torch
import torch.nn as nn
import os
import ast
import inspect

PADDING_MODES = ["zeros", "reflect", "replicate", "circular"]
NONLINEARITY_TYPES = ["relu", "leaky_relu", "selu", "tanh", "linear", "sigmoid"]
INIT_MODES = ["fan_in", "fan_out"]
LAYER_NORM_TYPES = ["None", "BatchNorm", "LayerNorm"]

# List of weight initialization methods
WEIGHT_INIT_FUNCTIONS_LIST = [
    'default',
    'normal',
    'uniform',
    'xavier_normal',
    'xavier_uniform',
    'kaiming_normal',
    'kaiming_uniform',
    'orthogonal',
]

# List of bias initialization methods
BIAS_INIT_FUNCTIONS_LIST = [
    'default',
    'zeros',
    'ones',
    'normal',
    'uniform',
]

# Global configuration lists at the top of the file
OPTIMIZER_TYPES = [
    "Adadelta",
    "Adagrad",
    "Adam",
    "AdamW",
    "SparseAdam",
    "Adamax",
    "ASGD",
    "NAdam",
    "RAdam",
    "RMSprop",
    "Rprop",
    "SGD"
]

ACTIVATION_FUNCTIONS = [
    "None",
    "ELU",
    "GELU",
    "GLU",
    "Hardshrink",
    "Hardsigmoid",
    "Hardswish",
    "Hardtanh",
    "LeakyReLU",
    "LogSigmoid",
    "MultiheadAttention",
    "PReLU",
    "ReLU",
    "ReLU6",
    "RReLU",
    "SELU",
    "CELU",
    "Sigmoid",
    "SiLU",
    "Softmax",
    "Softmax2d",
    "Softmin",
    "Softplus",
    "Softshrink",
    "Softsign",
    "Tanh",
    "Tanhshrink",
    "Threshold"
]

LOSS_FUNCTIONS = [
    "L1Loss",
    "MSELoss",
    "CrossEntropyLoss",
    "CTCLoss",
    "NLLLoss",
    "PoissonNLLLoss",
    "GaussianNLLLoss",
    "KLDivLoss",
    "BCELoss",
    "BCEWithLogitsLoss",
    "MarginRankingLoss",
    "HingeEmbeddingLoss",
    "MultiLabelMarginLoss",
    "HuberLoss",
    "SmoothL1Loss",
    "SoftMarginLoss",
    "MultiLabelSoftMarginLoss",
    "CosineEmbeddingLoss",
    "MultiMarginLoss",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss"
]

NORMALIZATION_TYPES = [
    "None",
    "BatchNorm",  # Will be automatically converted to BatchNorm1d/2d/3d
    "LayerNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "GroupNorm",
    "LocalResponseNorm"
]

PADDING_MODES = [
    "zeros",
    "reflect",
    "replicate",
    "circular"
]

POOLING_TYPES = [
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "AdaptiveMaxPool1d",
    "AdaptiveMaxPool2d",
    "AdaptiveMaxPool3d",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "FractionalMaxPool2d",
    "LPPool1d",
    "LPPool2d"
]

WEIGHT_INIT_METHODS = [
    "default",
    "normal",
    "uniform",
    "xavier_normal",
    "xavier_uniform",
    "kaiming_normal",
    "kaiming_uniform",
    "orthogonal",
    "sparse",
    "dirac",
    "zeros",
    "ones"
]

TRAINING_PARAMS = {
    # Core parameters
    "experiment_name": ("STRING", {
        "default": "training_experiment",
        "multiline": False,
    }),
    "batch_size": ("INT", {
        "default": 32,
        "min": 1,
        "max": 512,
        "step": 1
    }),
    "epochs": ("INT", {  # Changed from num_epochs for consistency
        "default": 10,
        "min": 1,
        "max": 1000,
        "step": 1
    }),

    # Optimizer settings
    "optimizer": (OPTIMIZER_TYPES, {  # Changed from optimizer_type for consistency
        "default": "Adam"
    }),
    "learning_rate": ("FLOAT", {
        "default": 0.001,
        "min": 0.000001,
        "max": 1.0,
        "step": 0.0001
    }),
    "weight_decay": ("FLOAT", {
        "default": 0.0001,
        "min": 0.0,
        "max": 0.1,
        "step": 0.0001
    }),
    "momentum": ("FLOAT", {
        "default": 0.9,
        "min": 0.0,
        "max": 1.0,
        "step": 0.1
    }),

    # Loss function settings
    "loss_function": (LOSS_FUNCTIONS, {
        "default": "CrossEntropyLoss"
    }),
    "reduction": (["mean", "sum", "none"], {
        "default": "mean"
    }),
    "weight_enabled": (["True", "False"], {
        "default": "False"
    }),
    "class_weights": ("STRING", {
        "default": "[]",
        "multiline": False,
        "placeholder": "e.g., [1.0, 2.0, 0.5] for 3 classes"
    }),
    "margin": ("FLOAT", {
        "default": 0.0,
        "min": -1.0,
        "max": 1.0,
        "step": 0.1
    }),

    # Learning rate scheduling
        "use_lr_scheduler": (["True", "False"], {
        "default": "True"
    }),
    "scheduler_type": (["StepLR", "ReduceLROnPlateau", "CosineAnnealingLR"], {
        "default": "ReduceLROnPlateau"
    }),
    "scheduler_step_size": ("INT", {
        "default": 3,
        "min": 1,
        "max": 100,
        "step": 1
    }),
    "scheduler_gamma": ("FLOAT", {
        "default": 0.1,
        "min": 0.01,
        "max": 1.0,
        "step": 0.01
    }),
    "min_lr": ("FLOAT", {  # Added this parameter
        "default": 0.000001,
        "min": 0.0000001,
        "max": 0.1,
        "step": 0.000001
    }),

    # Early stopping
    "use_early_stopping": (["True", "False"], {
        "default": "True"
    }),
    "patience": ("INT", {
        "default": 5,
        "min": 1,
        "max": 50,
        "step": 1
    }),
    "min_delta": ("FLOAT", {
        "default": 0.001,
        "min": 0.0001,
        "max": 0.1,
        "step": 0.0001
    }),
}

class NntShowModelInfo:
    """
    Node to display comprehensive model information, including layer architecture,
    parameters, initializations, and other configuration details.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("MODEL",),
                "show_details": (["Basic", "Detailed"], {"default": "Basic"}),
                "include_weights": (["True", "False"], {"default": "False"}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "model_info")
    FUNCTION = "show_model_info"
    OUTPUT_IS_LIST = (False, False)
    CATEGORY = "NNT Neural Network Toolkit/Models"

    def format_shape(self, tensor):
        """Helper function to format tensor shapes."""
        return 'x'.join(str(dim) for dim in tensor.shape)

    def format_memory(self, num_bytes):
        """Helper function to format memory usage."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if num_bytes < 1024:
                return f"{num_bytes:.2f} {unit}"
            num_bytes /= 1024
        return f"{num_bytes:.2f} TB"

    def get_activation_info(self, module):
        """Helper function to get activation function details."""
        info = []
        if hasattr(module, 'negative_slope'):
            info.append(f"negative_slope={module.negative_slope}")
        if hasattr(module, 'min_val'):
            info.append(f"min_val={module.min_val}")
        if hasattr(module, 'max_val'):
            info.append(f"max_val={module.max_val}")
        if hasattr(module, 'inplace'):
            info.append(f"inplace={module.inplace}")
        return ", ".join(info) if info else ""

    def get_normalization_info(self, module):
        """Helper function to get normalization layer details."""
        info = []
        if hasattr(module, 'eps'):
            info.append(f"eps={module.eps}")
        if hasattr(module, 'momentum'):
            info.append(f"momentum={module.momentum}")
        if hasattr(module, 'affine'):
            info.append(f"affine={module.affine}")
        if hasattr(module, 'track_running_stats'):
            info.append(f"track_running_stats={module.track_running_stats}")
        return ", ".join(info) if info else ""

    def show_model_info(self, MODEL, show_details, include_weights):
        import torch
        import torch.nn as nn
        
        # Initialize model information list
        model_info = []
        
        # General Model Statistics
        total_params = sum(p.numel() for p in MODEL.parameters())
        trainable_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
        total_memory = sum(p.nelement() * p.element_size() for p in MODEL.parameters())
        
        model_info.extend([
            "=== Model Summary ===",
            f"Total Parameters: {total_params:,}",
            f"Trainable Parameters: {trainable_params:,}",
            f"Memory Usage: {self.format_memory(total_memory)}",
            "\n=== Layer Architecture ===",
        ])

        module_idx = 0
        while module_idx < len(MODEL):
            module = MODEL[module_idx]
            
            # Base layer info
            layer_info = [f"\nLayer {module_idx + 1}:"]
            
            # Linear/Dense layers
            if isinstance(module, nn.Linear):
                layer_info.append(f"Type: Linear")
                layer_info.append(f"In Features: {module.in_features}")
                layer_info.append(f"Out Features: {module.out_features}")
                
                if show_details == "Detailed":
                    weight_init = getattr(module, 'weight_init_method', 'default')
                    bias_init = getattr(module, 'bias_init_method', 'default')
                    layer_info.extend([
                        f"Use Bias: {module.bias is not None}",
                        f"Weight Init: {weight_init}",
                        f"Bias Init: {bias_init}",
                        f"Weight Shape: {self.format_shape(module.weight.data)}",
                    ])
                    if include_weights == "True":
                        layer_info.extend([
                            f"Weight Stats: mean={module.weight.data.mean():.4f}, std={module.weight.data.std():.4f}",
                            f"Bias Stats: mean={module.bias.data.mean():.4f}, std={module.bias.data.std():.4f}" if module.bias is not None else "No Bias"
                        ])

            # Convolutional layers
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, 
                                  nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                layer_info.append(f"Type: {module.__class__.__name__}")
                layer_info.extend([
                    f"In Channels: {module.in_channels}",
                    f"Out Channels: {module.out_channels}",
                    f"Kernel Size: {module.kernel_size}",
                    f"Stride: {module.stride}",
                    f"Padding: {module.padding}"
                ])
                
                if show_details == "Detailed":
                    layer_info.extend([
                        f"Dilation: {module.dilation}",
                        f"Groups: {module.groups}",
                        f"Padding Mode: {module.padding_mode}",
                        f"Weight Shape: {self.format_shape(module.weight.data)}",
                    ])
                    if include_weights == "True":
                        layer_info.extend([
                            f"Weight Stats: mean={module.weight.data.mean():.4f}, std={module.weight.data.std():.4f}",
                            f"Bias Stats: mean={module.bias.data.mean():.4f}, std={module.bias.data.std():.4f}" if module.bias is not None else "No Bias"
                        ])

            # Recurrent layers
            elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
                layer_info.append(f"Type: {module.__class__.__name__}")
                layer_info.extend([
                    f"Input Size: {module.input_size}",
                    f"Hidden Size: {module.hidden_size}",
                    f"Num Layers: {module.num_layers}",
                    f"Bidirectional: {module.bidirectional}"
                ])
                
                if show_details == "Detailed":
                    layer_info.extend([
                        f"Batch First: {module.batch_first}",
                        f"Dropout: {module.dropout}",
                        f"Bias: {module.bias}"
                    ])

            # Normalization layers
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                                  nn.LayerNorm, nn.InstanceNorm1d, nn.InstanceNorm2d,
                                  nn.InstanceNorm3d, nn.GroupNorm)):
                layer_info.append(f"Type: {module.__class__.__name__}")
                norm_info = self.get_normalization_info(module)
                if norm_info:
                    layer_info.append(f"Parameters: {norm_info}")

            # Pooling layers
            elif isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
                                  nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
                                  nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
                                  nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)):
                layer_info.append(f"Type: {module.__class__.__name__}")
                if hasattr(module, 'kernel_size'):
                    layer_info.append(f"Kernel Size: {module.kernel_size}")
                if hasattr(module, 'stride'):
                    layer_info.append(f"Stride: {module.stride}")
                if hasattr(module, 'padding'):
                    layer_info.append(f"Padding: {module.padding}")

            # Activation functions
            elif isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.ELU, nn.SELU,
                                  nn.Tanh, nn.Sigmoid, nn.LogSigmoid, nn.Softmax, nn.LogSoftmax)):
                layer_info.append(f"Type: Activation ({module.__class__.__name__})")
                activation_info = self.get_activation_info(module)
                if activation_info:
                    layer_info.append(f"Parameters: {activation_info}")

            # Dropout layers
            elif isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                layer_info.append(f"Type: {module.__class__.__name__}")
                layer_info.append(f"Rate: {module.p}")
                if show_details == "Detailed":
                    layer_info.append(f"Training: {module.training}")

            model_info.append(" | ".join(layer_info))
            module_idx += 1

        # Return the model and the formatted text output
        return (MODEL, "\n".join(model_info))
    
class NntSaveModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("MODEL",),
                "filename": ("STRING", {"default": "model.pth"}),
                "model_path": ("STRING", {
                    "default": "",  # Empty for default path
                    "multiline": False,
                    "placeholder": "Leave empty for default path"
                }),
                "save_format": (["PyTorch Model", "State Dict", "TorchScript", "ONNX", "TorchScript Mobile", "Quantized", "SafeTensors"], {
                    "default": "PyTorch Model"
                }),
                "save_optimizer": (["True", "False"], {"default": "False"}),
                "optimizer": (OPTIMIZER_TYPES + ["None"], {
                    "default": "None"
                }),
                "quantization_type": (["dynamic", "static", "none"], {"default": "none"}),
                "quantization_bits": ("INT", {
                    "default": 8,
                    "min": 4,
                    "max": 32,
                    "step": 4
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("MODEL", "report")
    FUNCTION = "save_model"
    OUTPUT_NODE = True
    CATEGORY = "NNT Neural Network Toolkit/Models"

    @staticmethod
    def get_default_model_path():
        """Get default path for model storage"""
        import os
        import folder_paths

        if not "nnt_models" in folder_paths.folder_names_and_paths:
            model_path = os.path.join(folder_paths.models_dir, "nnt_models")
            folder_paths.add_model_folder_path("nnt_models", model_path)
            return model_path
        
        return os.path.join(folder_paths.models_dir, "nnt_models")

    def save_model(self, MODEL, filename, model_path, save_format, save_optimizer, 
                optimizer, quantization_type, quantization_bits):
        import torch
        from pathlib import Path
        import torch.quantization
        
        try:

            # Use default path if model_path is empty
            save_path = model_path if model_path.strip() else self.get_default_model_path()
            
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            
            # Get full file path
            file_path = os.path.join(save_path, filename)
            base_path = os.path.splitext(file_path)[0]            
            MODEL.eval()

            optimizer = None
            if save_optimizer == "True" and optimizer != "None":
                optimizer_class = getattr(torch.optim, optimizer)
                optimizer = optimizer_class(MODEL.parameters())

            if save_format == "PyTorch Model":
                save_dict = {'model': MODEL, 'model_state_dict': MODEL.state_dict()}
                if optimizer:
                    save_dict.update({
                        'optimizer_state_dict': optimizer.state_dict(),
                        'optimizer_type': optimizer
                    })
                save_dict['model_config'] = {
                    'input_shape': getattr(MODEL, 'input_shape', None),
                    'architecture': str(MODEL)
                }
                torch.save(save_dict, str(file_path))
                status_msg = f"Model saved to {file_path}"

            elif save_format == "State Dict":
                save_dict = {
                    'model_state_dict': MODEL.state_dict(),
                    'model_config': {
                        'input_shape': getattr(MODEL, 'input_shape', None),
                        'architecture': str(MODEL)
                    }
                }
                if optimizer:
                    save_dict.update({
                        'optimizer_state_dict': optimizer.state_dict(),
                        'optimizer_type': optimizer
                    })
                torch.save(save_dict, str(file_path))
                status_msg = f"State dict saved to {file_path}"

            elif save_format == "TorchScript":
                input_shape = getattr(MODEL, 'input_shape', None)
                if input_shape is None:
                    return (MODEL, "Error: Model has no input_shape attribute")
                example_input = torch.randn(1, *input_shape)
                scripted_model = torch.jit.script(MODEL)
                scripted_model.save(str(base_path.with_suffix('.pt')))
                status_msg = f"TorchScript model saved to {base_path.with_suffix('.pt')}"

            elif save_format == "ONNX":
                input_shape = getattr(MODEL, 'input_shape', None)
                if input_shape is None:
                    return (MODEL, "Error: Model has no input_shape attribute")
                dummy_input = torch.randn(1, *input_shape)
                torch.onnx.export(MODEL, dummy_input, str(base_path.with_suffix('.onnx')),
                                input_names=['input'],
                                output_names=['output'],
                                dynamic_axes={'input': {0: 'batch_size'},
                                            'output': {0: 'batch_size'}},
                                opset_version=11)
                status_msg = f"ONNX model saved to {base_path.with_suffix('.onnx')}"

            elif save_format == "TorchScript Mobile":
                input_shape = getattr(MODEL, 'input_shape', None)
                if input_shape is None:
                    return (MODEL, "Error: Model has no input_shape attribute")
                example_input = torch.randn(1, *input_shape)
                scripted_model = torch.jit.script(MODEL)
                scripted_model._save_for_lite_interpreter(str(base_path.with_suffix('.ptl')))
                status_msg = f"Mobile TorchScript saved to {base_path.with_suffix('.ptl')}"
            
            elif save_format == "SafeTensors":
                from safetensors.torch import save_file
                state_dict = MODEL.state_dict()
                # Save metadata separately as PyTorch model
                metadata = {
                    'model_config': {
                        'input_shape': getattr(MODEL, 'input_shape', None),
                        'architecture': str(MODEL)
                    }
                }
                torch.save(metadata, str(base_path.with_suffix('.config')))
                # Save tensors only
                save_file(state_dict, str(base_path.with_suffix('.safetensors')))
                status_msg = f"SafeTensors model saved to {base_path.with_suffix('.safetensors')}"
                
            elif save_format == "Quantized":
                if quantization_type == "dynamic":
                    quantized_model = torch.quantization.quantize_dynamic(
                        MODEL, {torch.nn.Linear, torch.nn.Conv2d}, 
                        dtype=torch.qint8
                    )
                elif quantization_type == "static":
                    MODEL.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                    torch.quantization.prepare(MODEL, inplace=True)
                    input_shape = getattr(MODEL, 'input_shape', None)
                    if input_shape is None:
                        return (MODEL, "Error: Model has no input_shape attribute")
                    example_input = torch.randn(1, *input_shape)
                    MODEL(example_input)  # Calibration
                    quantized_model = torch.quantization.convert(MODEL, inplace=False)
                else:
                    return (MODEL, "Invalid quantization type")

                torch.save(quantized_model.state_dict(), str(base_path.with_suffix('.quantized.pth')))
                status_msg = f"Quantized model saved to {base_path.with_suffix('.quantized.pth')}"

            if not file_path.exists() and not base_path.with_suffix('.pt').exists() and \
                not base_path.with_suffix('.onnx').exists() and not base_path.with_suffix('.ptl').exists() and \
                not base_path.with_suffix('.quantized.pth').exists():
                return (MODEL, f"Error: Failed to verify saved file")

            return (MODEL, status_msg)

        except Exception as e:
            return (MODEL, f"Error saving model: {str(e)}")

# New utility class for model analysis
class NntAnalyzeModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("MODEL",),
                "input_shape": ("STRING", {
                    "default": "[3, 32, 32]",
                    "multiline": False,
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 128,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "analyze_model"
    OUTPUT_NODE = True
    CATEGORY = "NNT Neural Network Toolkit/Models"

    def analyze_model(self, MODEL, input_shape, batch_size):
        try:
            import torch
            import numpy as np

            # Move model to CPU for analysis to avoid device mismatch
            MODEL = MODEL.cpu()
            
            # Parse input shape
            try:
                shape = eval(input_shape)
                if not isinstance(shape, (list, tuple)):
                    raise ValueError("Input shape must be a list")
            except Exception as e:
                return (f"Error parsing input shape: {str(e)}",)

            # Create sample input (on CPU)
            sample_input = torch.randn(batch_size, *shape)
            model = MODEL

            # Collect layer info
            layer_info = []
            total_params = 0
            trainable_params = 0
            
            for idx, (name, module) in enumerate(model.named_modules()):
                if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
                    params = sum(p.numel() for p in module.parameters())
                    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                    total_params += params
                    trainable_params += trainable
                    
                    if isinstance(module, nn.Linear):
                        layer_info.append(
                            f"Layer {idx}: Type=Linear, "
                            f"In Features={module.in_features}, "
                            f"Out Features={module.out_features}, "
                            f"Use Bias={module.bias is not None}, "
                            f"Weight Init=default, Bias Init=default"
                        )
                    elif isinstance(module, nn.Conv2d):
                        layer_info.append(
                            f"Layer {idx}: Type=Conv2d, "
                            f"In Channels={module.in_channels}, "
                            f"Out Channels={module.out_channels}, "
                            f"Kernel Size={module.kernel_size}, "
                            f"Stride={module.stride}, "
                            f"Padding={module.padding}"
                        )
                    elif isinstance(module, nn.BatchNorm2d):
                        layer_info.append(
                            f"Layer {idx}: Type=BatchNorm2d, "
                            f"Num Features={module.num_features}"
                        )
                else:
                    layer_info.append(f"Layer {idx}: Activation={module.__class__.__name__}")

            # Memory analysis
            memory_params = sum(p.nelement() * p.element_size() for p in model.parameters())
            memory_buffers = sum(b.nelement() * b.element_size() for b in model.buffers())
            memory_total = memory_params + memory_buffers

            output = [
                f"Model Analysis for input shape {shape}:",
                f"Total Parameters: {total_params:,}",
                f"Trainable Parameters: {trainable_params:,}",
                f"Memory Usage:",
                f"  Parameters: {memory_params / 1024 / 1024:.2f} MB",
                f"  Buffers: {memory_buffers / 1024 / 1024:.2f} MB",
                f"  Total: {memory_total / 1024 / 1024:.2f} MB",
                "",
                "Model Architecture:",
                *layer_info
            ]

            return ("\n".join(output),)

        except Exception as e:
            return (f"Error analyzing model: {str(e)}",)
    
#OPTIMIZER_TYPES = [
#    "Adam", "SGD", "RMSprop", "AdamW", 
#    "Adadelta", "Adagrad", "Adamax", "NAdam",
#    "RAdam"
#]
#
#LOSS_FUNCTIONS = [
#    "MSELoss", "CrossEntropyLoss", "BCELoss", 
#    "L1Loss", "SmoothL1Loss", "BCEWithLogitsLoss",
#    "HuberLoss", "KLDivLoss"
#]

class NntTrainModel:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "MODEL": ("MODEL",),
            "training_data": ("TENSOR",),
            "target_data": ("TENSOR",),
            
            **TRAINING_PARAMS,  # Include all training parameters
            # Add new input reshaping parameters
            "reshape_input": (["True", "False"], {
                "default": "False"
            }),
            "input_reshape_dim": ("STRING", {
                "default": "[-1]",
                "multiline": False,
                "placeholder": "e.g., [-1, 784] or [-1, 28, 28]"
            }),
            "flatten_input": (["True", "False"], {
                "default": "False"
            }),
        }
        
        optional = {
            "hyperparameters": ("DICT",)
        }
        
        return {
            "required": required,
            "optional": optional
        }

    RETURN_TYPES = ("MODEL", "STRING", "DICT")
    RETURN_NAMES = ("model", "training_log", "metrics")
    FUNCTION = "train_model"
    OUTPUT_NODE = True
    CATEGORY = "NNT Neural Network Toolkit/Models"

    def create_loss_function(self, loss_type, reduction, weight_enabled, class_weights, margin, device):
        import torch.nn as nn
        import torch
        
        try:
            # Move weights to the correct device if they exist
            weights = None
            if weight_enabled == "True":
                try:
                    weights = torch.tensor(eval(class_weights)).to(device)
                except:
                    print("Warning: Could not parse class weights, using None")

            if loss_type == "CrossEntropyLoss":
                return nn.CrossEntropyLoss(weight=weights, reduction=reduction).to(device)
            elif loss_type == "NLLLoss":
                return nn.NLLLoss(weight=weights, reduction=reduction).to(device)
            elif loss_type == "MSELoss":
                return nn.MSELoss(reduction=reduction).to(device)
            elif loss_type == "BCELoss":
                return nn.BCELoss(weight=weights, reduction=reduction).to(device)
            elif loss_type == "BCEWithLogitsLoss":
                return nn.BCEWithLogitsLoss(weight=weights, reduction=reduction).to(device)
            elif loss_type == "KLDivLoss":
                return nn.KLDivLoss(reduction=reduction).to(device)
            elif loss_type == "CosineEmbeddingLoss":
                return nn.CosineEmbeddingLoss(margin=margin, reduction=reduction).to(device)
            elif loss_type == "CTCLoss":
                return nn.CTCLoss(reduction=reduction, zero_infinity=True).to(device)
            elif loss_type == "TripletMarginLoss":
                return nn.TripletMarginLoss(margin=margin, reduction=reduction).to(device)
            else:
                return nn.MSELoss(reduction=reduction).to(device)
        except Exception as e:
            print(f"Error creating loss function: {str(e)}")
            return nn.MSELoss(reduction=reduction).to(device)
    
    def train_model(self, MODEL, training_data, target_data, experiment_name="training_experiment",
                    batch_size=32, epochs=10, optimizer="Adam", learning_rate=0.001, 
                    weight_decay=0.0001, momentum=0.9, loss_function="CrossEntropyLoss",
                    reduction="mean", weight_enabled="False", class_weights="[]", margin=0.0,
                    use_lr_scheduler="False", scheduler_type="StepLR", scheduler_step_size=10,
                    scheduler_gamma=0.1, min_lr=0.000001, use_early_stopping="True", patience=5, min_delta=0.001,
                    reshape_input="True", input_reshape_dim="[-1]", flatten_input="True",
                    hyperparameters=None):
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            import ast
            from time import time
            import logging

            with torch.inference_mode(False), torch.set_grad_enabled(True):
                # Logging setup
                logger = logging.getLogger(experiment_name)
                logger.setLevel(logging.INFO)
                if not logger.handlers:
                    handler = logging.StreamHandler()
                    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
                    logger.addHandler(handler)

                logger.info("Starting training process...")

                # Override parameters with hyperparameters if provided
                if hyperparameters is not None:
                    locals().update(hyperparameters.get('training', {}))

                # Prepare input data
                if reshape_input == "True":
                    try:
                        reshape_dims = ast.literal_eval(input_reshape_dim)
                        training_data = training_data.reshape(*reshape_dims)
                    except Exception as e:
                        logger.warning(f"Failed to reshape input: {str(e)}")
                elif flatten_input == "True":
                    training_data = training_data.reshape(training_data.shape[0], -1)

                # Ensure model is in training mode
                MODEL.train()

                # Device configuration
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                MODEL = MODEL.to(device)

                # Move data to device
                training_data = training_data.to(device)
                target_data = target_data.to(device)

                ## Prepare data
                training_data = training_data.to(device)
                target_data = target_data.to(device)

                # Reshape and validate data
                training_data, target_data = self._prepare_data_for_training(
                    training_data, target_data,
                    reshape_input, input_reshape_dim, flatten_input
                )

                # Prepare target data based on loss function type
                if loss_function == "MSELoss":
                    # For regression, just ensure tensors are float
                    training_data = training_data.float()
                    target_data = target_data.float()
                elif loss_function in ["CrossEntropyLoss", "NLLLoss"]:
                    # Classification task - keep targets as long
                    training_data = training_data.float()
                    target_data = target_data.squeeze() #.long()  # Add squeeze() to handle (N,1) shape
                elif loss_function == "CTCLoss":
                    # For CTC Loss, we need input_lengths and target_lengths
                    input_lengths = torch.full((training_data.size(0),), training_data.size(1), dtype=torch.long)
                    target_lengths = torch.full((target_data.size(0),), target_data.size(1), dtype=torch.long)
                    input_lengths = input_lengths.to(device)
                    target_lengths = target_lengths.to(device)
                else:
                    # Default to float tensors for other loss types
                    training_data = training_data.float()
                    target_data = target_data.float()

                # Create dataset and dataloader
                train_dataset = TensorDataset(training_data, target_data)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                # Create loss function
                criterion = self.create_loss_function(
                    loss_function, reduction, weight_enabled,
                    class_weights, margin, device
                )

                # Setup optimizer
                optimizer_class = getattr(torch.optim, optimizer)
                optimizer_kwargs = {'lr': learning_rate, 'weight_decay': weight_decay}
                if optimizer == "SGD":
                    optimizer_kwargs['momentum'] = momentum
                optimizer_instance = optimizer_class(MODEL.parameters(), **optimizer_kwargs)

                # Learning rate scheduler setup
                scheduler = None
                if use_lr_scheduler == "True":
                    if scheduler_type == "StepLR":
                        scheduler = optim.lr_scheduler.StepLR(
                            optimizer_instance, 
                            step_size=scheduler_step_size, 
                            gamma=scheduler_gamma
                        )
                    elif scheduler_type == "ReduceLROnPlateau":
                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer_instance,
                            mode='min',
                            factor=scheduler_gamma,
                            patience=scheduler_step_size,
                            min_lr=min_lr
                        )
                    elif scheduler_type == "CosineAnnealingLR":
                        scheduler = optim.lr_scheduler.CosineAnnealingLR(
                            optimizer_instance,
                            T_max=epochs
                        )

                # Metrics tracking
                metrics = {
                    "loss": [],
                    "accuracy": [],
                    "learning_rates": [],
                    "batch_losses": [],
                    "epoch_times": [],
                    "best_loss": float('inf'),
                    "best_accuracy": 0.0 if loss_function != "MSELoss" else None,
                    "total_time": 0.0
                }

                # Training epochs
                best_loss = float('inf')
                early_stopping_counter = 0
                epoch_logs = []
                total_start_time = time()

                for epoch in range(epochs):
                    epoch_start_time = time()
                    running_loss = 0.0
                    correct = 0
                    total = 0

                    current_lr = optimizer_instance.param_groups[0]['lr']
                    metrics["learning_rates"].append(current_lr)

                    for batch_idx, (inputs, labels) in enumerate(train_loader):
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # Forward pass
                        outputs = MODEL(inputs)
                        loss = criterion(outputs, labels)

                        # Backward pass and optimize
                        optimizer_instance.zero_grad()
                        loss.backward(retain_graph=True)
                        optimizer_instance.step()

                        # Track metrics
                        batch_loss = loss.item()
                        metrics["batch_losses"].append(batch_loss)
                        running_loss += batch_loss

                        # Track accuracy for classification tasks
                        if loss_function != "MSELoss":
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()

                    # Calculate epoch metrics
                    epoch_loss = running_loss / len(train_loader)
                    metrics["loss"].append(epoch_loss)

                    if epoch_loss < metrics["best_loss"]:
                        metrics["best_loss"] = epoch_loss

                    if loss_function != "MSELoss":
                        epoch_accuracy = 100 * correct / total if total > 0 else 0
                        metrics["accuracy"].append(epoch_accuracy)
                        if epoch_accuracy > metrics["best_accuracy"]:
                            metrics["best_accuracy"] = epoch_accuracy
                        epoch_log = f"Epoch {epoch+1:3d}/{epochs}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_accuracy:.2f}%"
                    else:
                        epoch_log = f"Epoch {epoch+1:3d}/{epochs}: MSE Loss = {epoch_loss:.4f}"

                    # Track epoch time
                    epoch_time = time() - epoch_start_time
                    metrics["epoch_times"].append(epoch_time)

                    epoch_logs.append(epoch_log)
                    logger.info(f"{epoch_log} (Time: {epoch_time:.2f}s, LR: {current_lr:.6f})")

                    # Learning rate scheduler step
                    if scheduler is not None:
                        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            scheduler.step(epoch_loss)
                        else:
                            scheduler.step()

                    # Early stopping check
                    if use_early_stopping == "True":
                        if epoch_loss < best_loss - float(min_delta):
                            best_loss = epoch_loss
                            early_stopping_counter = 0
                        else:
                            early_stopping_counter += 1
                            if early_stopping_counter >= int(patience):
                                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                                break

                # Record total training time
                metrics["total_time"] = time() - total_start_time

                # Create training summary
                training_summary = (
                    f"Training Results ({metrics['total_time']:.2f}s):\n" +
                    "-" * 50 + "\n" +
                    "\n".join(epoch_logs) + "\n" +
                    "-" * 50 + "\n" +
                    f"Final Loss: {metrics['loss'][-1]:.4f}" +
                    (f", Best Loss: {metrics['best_loss']:.4f}\n" if metrics['best_loss'] != float('inf') else "\n") +
                    (f"Final Accuracy: {metrics['accuracy'][-1]:.2f}%, Best Accuracy: {metrics['best_accuracy']:.2f}%\n" 
                    if metrics['accuracy'] else "") +
                    f"Final Learning Rate: {metrics['learning_rates'][-1]:.6f}"
                )

                return MODEL, training_summary, metrics

        except Exception as e:
            import traceback
            error_msg = f"Error during training: {str(e)}\n{traceback.format_exc()}"
            empty_metrics = {"loss": [], "accuracy": [], "learning_rates": [], 
                            "batch_losses": [], "epoch_times": [], 
                            "best_loss": float('inf'), "best_accuracy": 0.0,
                            "total_time": 0.0}
            return MODEL, error_msg, empty_metrics

    def _prepare_data_for_training(self, training_data, target_data, reshape_input, input_reshape_dim, flatten_input):
        """Prepare and validate training and target data shapes"""
        import ast
        with torch.inference_mode(False), torch.set_grad_enabled(True):
            # Print initial shapes for debugging
            print(f"Initial shapes - Training: {training_data.shape}, Target: {target_data.shape}")
            
            # Handle input reshaping
            if reshape_input == "True":
                try:
                    reshape_dims = ast.literal_eval(input_reshape_dim)
                    training_data = training_data.reshape(*reshape_dims)
                except Exception as e:
                    print(f"Failed to reshape input using {input_reshape_dim}: {str(e)}")
            elif flatten_input == "True":
                training_data = training_data.reshape(training_data.shape[0], -1)
            
            # Ensure target data is properly shaped
            if len(target_data.shape) == 1:
                target_data = target_data.reshape(-1, 1)
            
            # Validate sizes match on first dimension
            if training_data.shape[0] != target_data.shape[0]:
                raise ValueError(f"Number of samples mismatch: training_data has {training_data.shape[0]} samples but target_data has {target_data.shape[0]}")
            
            print(f"Final shapes - Training: {training_data.shape}, Target: {target_data.shape}")
            
            return training_data, target_data

class NntEvaluatePredictions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("MODEL",),
                "input_data": ("TENSOR",),
                "target_data": ("TENSOR",),
                "task_type": (["classification", "regression"], {
                    "default": "classification"
                }),
                "num_classes": ("INT", {
                    "default": 10,
                    "min": 2,
                    "max": 1000,
                    "step": 1,
                })
            }
        }

    RETURN_TYPES = ("DICT", "STRING")
    RETURN_NAMES = ("metrics", "report")
    FUNCTION = "evaluate_predictions"
    OUTPUT_NODE = True
    CATEGORY = "NNT Neural Network Toolkit/Models"

    def evaluate_predictions(self, MODEL, input_data, target_data, task_type, num_classes):
        try:
            import torch
            import numpy as np
            from sklearn.metrics import confusion_matrix
            
            # Add this context manager similar to NntTrainModel
            with torch.inference_mode(False), torch.set_grad_enabled(True):
                # Determine device and move data
                device = next(MODEL.parameters()).device
                
                # For classification, keep targets as long integers
                if task_type == "classification":
                    input_data = input_data.float().to(device)  
                    target_data = target_data.squeeze().long().to(device)
                else:
                    input_data = input_data.float().to(device)
                    target_data = target_data.float().to(device)

                # Generate predictions
                MODEL.eval()

                with torch.no_grad():
                    predictions = MODEL(input_data)

                    # Ensure predictions have correct shape
                    if len(predictions.shape) == 1:
                        predictions = predictions.unsqueeze(1)

                    # Move to CPU for metrics calculation
                    predictions = predictions.cpu()
                    target_data = target_data.cpu()

                    if task_type == "classification":
                        # Get class predictions
                        pred_classes = torch.argmax(predictions, dim=1)
                        softmax_preds = torch.softmax(predictions, dim=1)
                        
                        # Ensure target_data matches prediction shape
                        if len(target_data.shape) == 1:
                            target_data = target_data.view(-1)
                        
                        # Calculate metrics
                        correct = (pred_classes == target_data).float()
                        accuracy = correct.mean().item()
                        
                        # Calculate per-class accuracy
                        cm = confusion_matrix(target_data, pred_classes)
                        per_class_acc = cm.diagonal() / cm.sum(axis=1)
                        
                        # Calculate confidence scores
                        confidences = softmax_preds.max(dim=1)[0]

                        metrics_dict = {
                            "accuracy": accuracy,
                            "per_class_accuracy": per_class_acc.tolist(),
                            "predictions": pred_classes.tolist(),
                            "true_labels": target_data.tolist(),
                            "confidences": confidences.tolist(),
                            "confusion_matrix": cm.tolist(),
                        }

                        # Generate report
                        report_lines = [
                            "=== Classification Results ===",
                            f"Overall Accuracy: {accuracy:.2%}",
                            f"Samples Evaluated: {len(target_data)}",
                            "\nPer-Class Accuracy:",
                        ]
                        for i, acc in enumerate(per_class_acc):
                            report_lines.append(f"Class {i}: {acc:.2%}")

                    else:  # regression
                        predictions = predictions.squeeze()
                        target_data = target_data.squeeze()
                        
                        # Ensure matching shapes
                        if predictions.shape != target_data.shape:
                            predictions = predictions.reshape(-1)
                            target_data = target_data.reshape(-1)
                        
                        # Calculate regression metrics
                        mse = torch.mean((predictions - target_data) ** 2).item()
                        mae = torch.mean(torch.abs(predictions - target_data)).item()
                        
                        metrics_dict = {
                            "predictions": predictions.tolist(),
                            "true_values": target_data.tolist(),
                            "mse": mse,
                            "mae": mae,
                        }

                        report_lines = [
                            "=== Regression Results ===",
                            f"Mean Squared Error: {mse:.4f}",
                            f"Mean Absolute Error: {mae:.4f}",
                            f"Samples Evaluated: {len(target_data)}"
                        ]

                    return (metrics_dict, "\n".join(report_lines))

        except Exception as e:
            import traceback
            error_msg = f"Error analyzing predictions: {str(e)}\n{traceback.format_exc()}"
            return ({}, error_msg)


class NntVisualizeTrainingMetrics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "metrics": ("DICT",),
                "image_width": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 64
                }),
                "image_height": ("INT", {
                    "default": 768,
                    "min": 256,
                    "max": 4096,
                    "step": 64
                }),
                "plot_type": ([
                    "loss", 
                    "accuracy", 
                    "combined",
                    "learning_rate",
                    "all_metrics",
                    "loss_with_lr"
                ], {
                    "default": "combined"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "visualize_metrics"
    OUTPUT_NODE = True
    CATEGORY = "NNT Neural Network Toolkit/Models"

    def visualize_metrics(self, metrics, image_width, image_height, plot_type):
        try:
            import torch
            import matplotlib.pyplot as plt
            import numpy as np
            from PIL import Image
            import io
            
            # Setup figure with the right size
            dpi = 100
            fig_width = image_width / dpi
            fig_height = image_height / dpi
            
            if plot_type == "all_metrics":
                # Create multiple subplots for all metrics
                fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height), dpi=dpi)
                axes = axes.flatten()
                
                # Loss plot
                if "loss" in metrics and metrics["loss"]:
                    axes[0].plot(metrics["loss"], 'r-', label='Loss')
                    axes[0].set_title('Training Loss')
                    axes[0].set_xlabel('Epoch')
                    axes[0].set_ylabel('Loss')
                    axes[0].grid(True)
                    axes[0].legend()
                
                # Accuracy plot
                if "accuracy" in metrics and metrics["accuracy"]:
                    axes[1].plot(metrics["accuracy"], 'b-', label='Accuracy')
                    axes[1].set_title('Training Accuracy')
                    axes[1].set_xlabel('Epoch')
                    axes[1].set_ylabel('Accuracy')
                    axes[1].grid(True)
                    axes[1].legend()
                
                # Learning rate plot
                if "learning_rates" in metrics and metrics["learning_rates"]:
                    axes[2].plot(metrics["learning_rates"], 'g-', label='Learning Rate')
                    axes[2].set_title('Learning Rate')
                    axes[2].set_xlabel('Epoch')
                    axes[2].set_ylabel('Learning Rate')
                    axes[2].grid(True)
                    axes[2].legend()
                
                # Batch losses if available
                if "batch_losses" in metrics and metrics["batch_losses"]:
                    axes[3].plot(metrics["batch_losses"], 'y-', label='Batch Losses')
                    axes[3].set_title('Batch Losses')
                    axes[3].set_xlabel('Batch')
                    axes[3].set_ylabel('Loss')
                    axes[3].grid(True)
                    axes[3].legend()
                
            elif plot_type == "loss_with_lr":
                # Create figure with two y-axes
                fig, ax1 = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
                ax2 = ax1.twinx()
                
                # Plot loss
                if "loss" in metrics and metrics["loss"]:
                    ax1.plot(metrics["loss"], 'r-', label='Loss')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss', color='r')
                    ax1.tick_params(axis='y', labelcolor='r')
                
                # Plot learning rate
                if "learning_rates" in metrics and metrics["learning_rates"]:
                    ax2.plot(metrics["learning_rates"], 'b-', label='Learning Rate')
                    ax2.set_ylabel('Learning Rate', color='b')
                    ax2.tick_params(axis='y', labelcolor='b')
                
                plt.title('Loss and Learning Rate')
                
            elif plot_type == "learning_rate":
                fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
                if "learning_rates" in metrics and metrics["learning_rates"]:
                    ax.plot(metrics["learning_rates"], 'g-')
                    ax.set_title('Learning Rate Schedule')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Learning Rate')
                    ax.grid(True)
                
            else:
                # Original combined or single metric plots
                if plot_type == "combined":
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height), dpi=dpi)
                    
                    if "loss" in metrics and metrics["loss"]:
                        ax1.plot(metrics["loss"], 'r-', label='Loss')
                        ax1.set_title('Training Loss')
                        ax1.set_xlabel('Epoch')
                        ax1.set_ylabel('Loss')
                        ax1.grid(True)
                        ax1.legend()
                    
                    if "accuracy" in metrics and metrics["accuracy"]:
                        ax2.plot(metrics["accuracy"], 'b-', label='Accuracy')
                        ax2.set_title('Training Accuracy')
                        ax2.set_xlabel('Epoch')
                        ax2.set_ylabel('Accuracy')
                        ax2.grid(True)
                        ax2.legend()
                else:
                    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
                    metric_name = plot_type
                    if metric_name in metrics and metrics[metric_name]:
                        ax.plot(metrics[metric_name])
                        ax.set_title(f'Training {metric_name.title()}')
                        ax.set_xlabel('Epoch')
                        ax.set_ylabel(metric_name.title())
                        ax.grid(True)

            plt.tight_layout()
            
            # Convert plot to tensor
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            
            # Convert to PIL Image and then to tensor
            image = Image.open(buf).convert('RGB')
            image_tensor = torch.tensor(np.array(image).astype(np.float32) / 255.0)
            image_tensor = torch.unsqueeze(image_tensor, 0)
            
            plt.close(fig)
            buf.close()
            
            # Create metrics summary
            summary = self.create_metrics_summary(metrics)
            
            return (image_tensor, summary)
            
        except Exception as e:
            import traceback
            error_msg = f"Error visualizing metrics: {str(e)}\n{traceback.format_exc()}"
            empty_tensor = torch.zeros((1, image_height, image_width, 3), dtype=torch.float32)
            return (empty_tensor, error_msg)

    def create_metrics_summary(self, metrics):
        """Create a text summary of the metrics"""
        summary_lines = ["Training Metrics Summary:", "-" * 25]
        
        if "loss" in metrics and metrics["loss"]:
            final_loss = metrics["loss"][-1]
            best_loss = min(metrics["loss"])
            summary_lines.extend([
                f"Final Loss: {final_loss:.4f}",
                f"Best Loss: {best_loss:.4f}"
            ])
            
        if "accuracy" in metrics and metrics["accuracy"]:
            final_acc = metrics["accuracy"][-1]
            best_acc = max(metrics["accuracy"])
            summary_lines.extend([
                f"Final Accuracy: {final_acc:.2f}%",
                f"Best Accuracy: {best_acc:.2f}%"
            ])
            
        if "learning_rates" in metrics and metrics["learning_rates"]:
            initial_lr = metrics["learning_rates"][0]
            final_lr = metrics["learning_rates"][-1]
            summary_lines.extend([
                f"Initial LR: {initial_lr:.6f}",
                f"Final LR: {final_lr:.6f}"
            ])
            
        return "\n".join(summary_lines)
    
    #        plot_image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

import shap
import matplotlib.pyplot as plt
import torch
import numpy as np
from io import BytesIO
from PIL import Image

class NntSHAPSummaryNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),  # Expecting a trained PyTorch model
                "X_train_sample": ("TENSOR",),  # Training sample data
                "X_test_sample": ("TENSOR",),  # Test sample data to explain
                "plot_type": (["dot", "bar", "violin"], {"default": "dot"}),  # SHAP plot type
            },
            "optional": {
                "background_sample_size": ("INT", {"default": 100, "min": 1, "max": 10000}),
            },
        }

    RETURN_TYPES = ("STRING", "TENSOR")
    RETURN_NAMES = ("text_report", "shap_plot_tensor")
    FUNCTION = "generate_shap_summary"
    OUTPUT_NODE = True
    CATEGORY = "NNT Neural Network Toolkit/Models"

    def generate_shap_summary(self, model, X_train_sample, X_test_sample, plot_type="dot", background_sample_size=100):
        try:
            # Validate inputs
            if X_train_sample.shape[0] < background_sample_size:
                raise ValueError(f"Background sample size {background_sample_size} exceeds training data size {X_train_sample.shape[0]}.")

            # Select a background sample for SHAP explanation
            background_sample = X_train_sample[:background_sample_size]

            # Use SHAP KernelExplainer
            explainer = shap.KernelExplainer(model, background_sample)
            shap_values = explainer.shap_values(X_test_sample)

            # Create SHAP summary plot
            plt.figure(figsize=(16, 8))
            shap.summary_plot(shap_values[1], X_train_sample, plot_type=plot_type, show=False)
            
            # Save plot to an image tensor
            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            image = Image.open(buf)
            image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
            buf.close()
            plt.close()

            # Generate text report
            feature_importance = np.abs(shap_values[1]).mean(axis=0)
            sorted_indices = np.argsort(feature_importance)[::-1]
            top_features = sorted_indices[:10]
            report_lines = ["Top 10 Features by Importance:"]
            for idx in top_features:
                report_lines.append(f"Feature {idx}: {feature_importance[idx]:.4f}")
            text_report = "\n".join(report_lines)

            return (text_report, image_tensor)
        except Exception as e:
            return (f"Error generating SHAP summary: {str(e)}", torch.zeros(3, 256, 256))  # Return empty tensor in case of error


class NntFineTuneModel:
    """
    Node for fine-tuning a neural network model. Assumes the model layers are already configured.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("MODEL",),
                "train_data": ("TENSOR",),
                "train_labels": ("TENSOR",),
                "val_data": ("TENSOR",),
                "val_labels": ("TENSOR",),
                "learning_rate": ("FLOAT", {
                    "default": 1e-4,
                    "min": 1e-8,
                    "max": 1.0,
                    "step": 1e-5
                }),
                "epochs": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "batch_size": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 1024,
                    "step": 1
                }),
                "loss_function": (LOSS_FUNCTIONS, {
                    "default": "CrossEntropyLoss"
                }),
                "optimizer": (OPTIMIZER_TYPES, {
                    "default": "Adam"
                }),
                "optimizer_params": ("STRING", {
                    "default": "{}",
                    "placeholder": "e.g., {'weight_decay': 1e-5}"
                }),
                "use_scheduler": (["True", "False"], {"default": "False"}),
                "scheduler": ([
                    "StepLR",
                    "ReduceLROnPlateau",
                    "ExponentialLR",
                    "CosineAnnealingLR"
                ], {"default": "StepLR"}),
                "scheduler_params": ("STRING", {
                    "default": "{}",
                    "placeholder": "e.g., {'step_size': 7, 'gamma': 0.1}"
                }),
                "early_stopping": (["True", "False"], {"default": "False"}),
                "early_stopping_patience": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
                "save_best_model": (["True", "False"], {"default": "True"}),
                "best_model_path": ("STRING", {
                    "default": "best_model.pth",
                    "placeholder": "Path to save the best model"
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("fine_tuned_model", "training_log")
    FUNCTION = "fine_tune_model"
    OUTPUT_NODE = True
    CATEGORY = "NNT Neural Network Toolkit/Models"

    def fine_tune_model(self, MODEL, train_data, train_labels, val_data, val_labels,
                        learning_rate, epochs, batch_size, loss_function, optimizer,
                        optimizer_params, use_scheduler, scheduler, scheduler_params,
                        early_stopping, early_stopping_patience, save_best_model, best_model_path):
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        import ast
        import copy

        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = MODEL.to(device)

            # Prepare datasets and dataloaders
            train_dataset = TensorDataset(train_data, train_labels)
            val_dataset = TensorDataset(val_data, val_labels)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            # Set up loss function
            criterion = getattr(nn, loss_function)()

            # Set up optimizer
            optimizer_params_dict = ast.literal_eval(optimizer_params)
            params = [param for param in model.parameters() if param.requires_grad]
            optimizer_class = getattr(optim, optimizer)
            optimizer_instance = optimizer_class(params, lr=learning_rate, **optimizer_params_dict)

            # Set up scheduler
            if use_scheduler == "True":
                scheduler_params_dict = ast.literal_eval(scheduler_params)
                scheduler_instance = getattr(optim.lr_scheduler, scheduler)(
                    optimizer_instance, **scheduler_params_dict
                )
            else:
                scheduler_instance = None

            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0
            training_log = []

            for epoch in range(epochs):
                train_loss = self.train_one_epoch(
                    model, train_loader, criterion, optimizer_instance, device
                )
                val_loss = self.validate(
                    model, val_loader, criterion, device
                )

                if use_scheduler == "True" and scheduler_instance is not None:
                    if scheduler == "ReduceLROnPlateau":
                        scheduler_instance.step(val_loss)
                    else:
                        scheduler_instance.step()

                log_message = f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                training_log.append(log_message)
                print(log_message)

                # Early stopping
                if early_stopping == "True":
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        if save_best_model == "True":
                            torch.save(model.state_dict(), best_model_path)
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            training_log.append("Early stopping triggered.")
                            break
                else:
                    if save_best_model == "True" and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), best_model_path)

            # Load the best model weights
            if save_best_model == "True" and os.path.exists(best_model_path):
                model.load_state_dict(torch.load(best_model_path))

            return (model, "\n".join(training_log))

        except Exception as e:
            error_message = f"Error during fine-tuning: {str(e)}"
            return (MODEL, error_message)

    def train_one_epoch(self, model, train_loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def validate(self, model, val_loader, criterion, device):
        model.eval()
        running_loss = 0.0
        #with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(val_loader.dataset)
        return epoch_loss

class NntEditModelLayers:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("MODEL",),
                "operation": (["Freeze", "Unfreeze", "Set weights and biases", "Prune", "Quantize"], {
                    "default": "Freeze"
                }),
                "parameter_type": (["weights", "biases", "both"], {
                    "default": "both"
                }),
                "layer_selection": (["All layers", "First N layers", "Last N layers", "Selected types"], {
                    "default": "All layers"
                }),
                "layer_types": (["Linear", "Conv2d", "BatchNorm2d", "All"], {
                    "default": "All"
                }),
                "num_layers": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "initialization": ([
                    "random", "zeros", "ones", "xavier_uniform", "xavier_normal",
                    "kaiming_uniform", "kaiming_normal", "orthogonal", "custom_value"
                ], {
                    "default": "kaiming_normal"
                }),
                "custom_value": ("FLOAT", {
                    "default": 0.0,
                    "min": -1e10,
                    "max": 1e10,
                    "step": 0.1
                }),
                "pruning_amount": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 0.9,
                    "step": 0.1
                }),
                "quantization_bits": ("INT", {
                    "default": 8,
                    "min": 4,
                    "max": 32,
                    "step": 4
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING", "DICT")
    RETURN_NAMES = ("edited_model", "info_message", "layer_stats")
    FUNCTION = "edit_model_layers"
    OUTPUT_NODE = True
    CATEGORY = "NNT Neural Network Toolkit/Models"

    def edit_model_layers(self, MODEL, operation, parameter_type, layer_selection, layer_types, 
                         num_layers, initialization, custom_value, pruning_amount, quantization_bits):
        try:
            import torch
            import torch.nn.init as init
            import torch.nn as nn

            model = MODEL
            info_message = []
            stats = {}

            all_layers = list(model.named_modules())[1:]
            
            if layer_types != "All":
                all_layers = [(name, layer) for name, layer in all_layers 
                            if layer.__class__.__name__ == layer_types]

            if layer_selection == "All layers":
                selected_layers = all_layers
            elif layer_selection == "First N layers":
                selected_layers = all_layers[:num_layers]
            elif layer_selection == "Last N layers":
                selected_layers = all_layers[-num_layers:]
            else:
                selected_layers = all_layers

            if operation == "Freeze":
                self.freeze_unfreeze_layers(selected_layers, parameter_type, True)
                info_message.append(f"Frozen {parameter_type} in layers: {[name for name, _ in selected_layers]}")

            elif operation == "Unfreeze":
                self.freeze_unfreeze_layers(selected_layers, parameter_type, False)
                info_message.append(f"Unfrozen {parameter_type} in layers: {[name for name, _ in selected_layers]}")

            elif operation == "Set weights and biases":
                stats = self.set_parameters(selected_layers, parameter_type, initialization, custom_value, init)
                info_message.append(f"Initialized {parameter_type} using {initialization}")

            elif operation == "Prune":
                stats = self.prune_layers(selected_layers, parameter_type, pruning_amount)
                info_message.append(f"Pruned {pruning_amount*100}% of {parameter_type}")

            elif operation == "Quantize":
                stats = self.quantize_layers(selected_layers, parameter_type, quantization_bits)
                info_message.append(f"Quantized {parameter_type} to {quantization_bits} bits")

            stats.update(self.get_layer_stats(selected_layers))
            return (model, "\n".join(info_message), stats)

        except Exception as e:
            return (MODEL, f"Error: {str(e)}", {})

    def freeze_unfreeze_layers(self, layers, parameter_type, freeze):
        for _, module in layers:
            for name, param in module.named_parameters():
                if (parameter_type == "weights" and "weight" in name) or \
                   (parameter_type == "biases" and "bias" in name) or \
                   parameter_type == "both":
                    param.requires_grad = not freeze

    def set_parameters(self, layers, parameter_type, algorithm, custom_value, init):
        stats = {'modified_params': 0}
        init_funcs = {
            'xavier_uniform': init.xavier_uniform_,
            'xavier_normal': init.xavier_normal_,
            'kaiming_uniform': init.kaiming_uniform_,
            'kaiming_normal': init.kaiming_normal_,
            'orthogonal': init.orthogonal_,
            'zeros': lambda x: init.constant_(x, 0),
            'ones': lambda x: init.constant_(x, 1),
            'custom_value': lambda x: init.constant_(x, custom_value),
            'random': init.normal_
        }

        for _, module in layers:
            for name, param in module.named_parameters():
                if (parameter_type == "weights" and "weight" in name) or \
                   (parameter_type == "biases" and "bias" in name) or \
                   parameter_type == "both":
                    init_funcs[algorithm](param)
                    stats['modified_params'] += param.numel()
        return stats

    def prune_layers(self, layers, parameter_type, amount):
        stats = {'pruned_params': 0, 'total_params': 0}
        for _, module in layers:
            for name, param in module.named_parameters():
                if (parameter_type == "weights" and "weight" in name) or \
                   (parameter_type == "biases" and "bias" in name) or \
                   parameter_type == "both":
                    mask = torch.ones_like(param.data, dtype=torch.bool)
                    threshold = torch.quantile(torch.abs(param.data), amount)
                    mask[torch.abs(param.data) < threshold] = False
                    param.data[~mask] = 0
                    stats['pruned_params'] += (~mask).sum().item()
                    stats['total_params'] += param.numel()
        return stats

    def quantize_layers(self, layers, parameter_type, bits):
        stats = {'quantized_params': 0}
        for _, module in layers:
            for name, param in module.named_parameters():
                if (parameter_type == "weights" and "weight" in name) or \
                   (parameter_type == "biases" and "bias" in name) or \
                   parameter_type == "both":
                    max_val = torch.max(torch.abs(param.data))
                    scale = (2 ** (bits - 1) - 1) / max_val
                    param.data = torch.round(param.data * scale) / scale
                    stats['quantized_params'] += param.numel()
        return stats

    def get_layer_stats(self, layers):
        stats = {
            'total_params': 0,
            'trainable_params': 0,
            'layer_types': {},
            'param_stats': {}
        }

        for name, module in layers:
            layer_type = module.__class__.__name__
            stats['layer_types'][layer_type] = stats['layer_types'].get(layer_type, 0) + 1
            
            for param_name, param in module.named_parameters():
                param_data = param.data.float()
                stats['total_params'] += param.numel()
                if param.requires_grad:
                    stats['trainable_params'] += param.numel()
                
                stats['param_stats'][f"{name}.{param_name}"] = {
                    'min': param_data.min().item(),
                    'max': param_data.max().item(),
                    'mean': param_data.mean().item(),
                    'std': param_data.std().item()
                }
        return stats
    
class NntMergeExtendModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "operation": (["Merge models", "Add layers"], {"default": "Merge models"}),
                "MODEL_A": ("MODEL",),
                # For merging
                "merge_method": (["weighted_average", "layer_fusion", "alternating"], {"default": "weighted_average"}),
                "weight_a": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "MODEL_B": ("MODEL",),
                "LAYER_STACK": ("LIST",),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    FUNCTION = "merge_extend_model"
    OUTPUT_NODE = True
    CATEGORY = "NNT Neural Network Toolkit/Models"

    def merge_extend_model(self, operation, MODEL_A, merge_method, weight_a, MODEL_B=None, LAYER_STACK=None):
        import torch
        import torch.nn as nn
        
        try:
            if operation == "Merge models":
                if MODEL_B is None:
                    return (MODEL_A, "Error: Second model required for merging")
                
                if not self.are_models_compatible(MODEL_A, MODEL_B):
                    return (MODEL_A, "Error: Models have incompatible architectures")
                
                merged_model = self.merge_models(MODEL_A, MODEL_B, merge_method, weight_a)
                info = f"Models merged using {merge_method} method"
                
            else:  # Add layers
                if LAYER_STACK is None:
                    return (MODEL_A, "Error: Layer stack required for adding layers")
                
                extended_model = self.extend_model(MODEL_A, LAYER_STACK)
                merged_model = extended_model
                info = f"Added {len(LAYER_STACK)} layers to model"
            
            return (merged_model, info)
            
        except Exception as e:
            return (MODEL_A, f"Error: {str(e)}")

    def are_models_compatible(self, model_a, model_b):
        """Check if models have compatible architectures"""
        def get_structure(model):
            return [(name, type(module), module._parameters.keys()) 
                    for name, module in model.named_modules()]
        
        struct_a = get_structure(model_a)
        struct_b = get_structure(model_b)
        
        return len(struct_a) == len(struct_b) and \
               all(ta == tb and ka == kb 
                   for (_, ta, ka), (_, tb, kb) in zip(struct_a, struct_b))

    def merge_models(self, model_a, model_b, method, weight_a):
        if method == "weighted_average":
            return self.weighted_average_merge(model_a, model_b, weight_a)
        elif method == "layer_fusion":
            return self.layer_fusion_merge(model_a, model_b)
        else:  # alternating
            return self.alternating_merge(model_a, model_b)

    def weighted_average_merge(self, model_a, model_b, weight_a):
        merged_model = type(model_a)(*model_a.__init_args__)
        weight_b = 1.0 - weight_a
        
        #with torch.no_grad():
        for (name_a, param_a), (name_b, param_b) in zip(
            model_a.named_parameters(), model_b.named_parameters()
        ):
            merged_param = weight_a * param_a + weight_b * param_b
            dict(merged_model.named_parameters())[name_a].copy_(merged_param)
        
        return merged_model

    def layer_fusion_merge(self, model_a, model_b):
        merged_model = type(model_a)(*model_a.__init_args__)
        
        #with torch.no_grad():
        for (name_a, param_a), (name_b, param_b) in zip(
            model_a.named_parameters(), model_b.named_parameters()
        ):
            merged_param = torch.max(param_a, param_b)
            dict(merged_model.named_parameters())[name_a].copy_(merged_param)
        
        return merged_model

    def alternating_merge(self, model_a, model_b):
        merged_model = type(model_a)(*model_a.__init_args__)
        
        #with torch.no_grad():
        params_merged = dict(merged_model.named_parameters())
        params_a = dict(model_a.named_parameters())
        params_b = dict(model_b.named_parameters())
        
        for i, name in enumerate(params_merged.keys()):
            source_params = params_a if i % 2 == 0 else params_b
            params_merged[name].copy_(source_params[name])
        
        return merged_model

    def extend_model(self, base_model, layer_stack):
        import torch.nn as nn
        
        def get_last_features(model):
            for module in reversed(list(model.modules())):
                if hasattr(module, 'out_features'):
                    return module.out_features
                if hasattr(module, 'out_channels'):
                    return module.out_channels
            raise ValueError("Could not determine output size of model")

        layers = list(base_model.children())
        prev_features = get_last_features(base_model)
        input_shape = getattr(base_model, 'input_shape', None)

        for layer_def in layer_stack:
            layer_type = layer_def['type']
            if layer_type == 'Linear':
                layers.append(
                    nn.Linear(
                        in_features=prev_features,
                        out_features=layer_def['num_nodes'],
                        bias=layer_def.get('use_bias', True)
                    )
                )
                prev_features = layer_def['num_nodes']
            elif layer_type == 'Conv2d':
                layers.append(
                    nn.Conv2d(
                        in_channels=prev_features,
                        out_channels=layer_def['out_channels'],
                        kernel_size=layer_def['kernel_size'],
                        stride=layer_def.get('stride', 1),
                        padding=layer_def.get('padding', 0)
                    )
                )
                prev_features = layer_def['out_channels']
            
            if layer_def.get('activation'):
                layers.append(getattr(nn, layer_def['activation'])())

        new_model = nn.Sequential(*layers)
        if input_shape is not None:
            new_model.input_shape = input_shape

        return new_model
    

class NntCompileModel:
    @classmethod
    
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["Compile", "Only create script"], {"default": "Compile"}),
                "LAYER_STACK": ("LIST",),
                "activation_function": (ACTIVATION_FUNCTIONS, {"default": "ReLU"}),
                "normalization": (NORMALIZATION_TYPES, {"default": "None"}),
                "padding_mode": (PADDING_MODES, {"default": "zeros"}),
                "weight_init": (WEIGHT_INIT_METHODS, {"default": "kaiming_normal"}),
                "activation_params": ("STRING", {
                    "default": "{}",
                    "placeholder": "e.g., {'alpha': 1.0, 'inplace': false}",
                    "multiline": True
                })
            },
            "optional": {
                "hyperparameters": ("DICT",)
            }
        }

    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("model", "report", "script")
    FUNCTION = "compile_model"
    CATEGORY = "NNT Neural Network Toolkit/Models"
    # Define activation map
    activation_map = {
        'ELU': lambda params: nn.ELU(alpha=params.get('alpha', 1.0)),
        'GELU': lambda params: nn.GELU(),
        'GLU': lambda params: nn.GLU(dim=params.get('dim', -1)),
        'Hardshrink': lambda params: nn.Hardshrink(lambd=params.get('lambd', 0.5)),
        'Hardsigmoid': lambda params: nn.Hardsigmoid(),
        'Hardswish': lambda params: nn.Hardswish(),
        'Hardtanh': lambda params: nn.Hardtanh(
            min_val=params.get('min_val', -1.0),
            max_val=params.get('max_val', 1.0)
        ),
        'LeakyReLU': lambda params: nn.LeakyReLU(negative_slope=params.get('negative_slope', 0.01)),
        'LogSigmoid': lambda params: nn.LogSigmoid(),
        'MultiheadAttention': lambda params: nn.MultiheadAttention(
            embed_dim=params.get('embed_dim', 512),
            num_heads=params.get('num_heads', 8),
            dropout=params.get('dropout', 0.0),
            batch_first=params.get('batch_first', True)
        ),
        'PReLU': lambda params: nn.PReLU(
            num_parameters=params.get('num_parameters', 1),
            init=params.get('init', 0.25)
        ),
        'ReLU': lambda params: nn.ReLU(inplace=params.get('inplace', False)),
        'ReLU6': lambda params: nn.ReLU6(inplace=params.get('inplace', False)),
        'RReLU': lambda params: nn.RReLU(
            lower=params.get('lower', 0.125),
            upper=params.get('upper', 0.3333),
            inplace=params.get('inplace', False)
        ),
        'SELU': lambda params: nn.SELU(inplace=params.get('inplace', False)),
        'CELU': lambda params: nn.CELU(
            alpha=params.get('alpha', 1.0),
            inplace=params.get('inplace', False)
        ),
        'Sigmoid': lambda params: nn.Sigmoid(),
        'SiLU': lambda params: nn.SiLU(inplace=params.get('inplace', False)),
        'Softmax': lambda params: nn.Softmax(dim=params.get('dim', -1)),
        'Softmax2d': lambda params: nn.Softmax2d(),
        'Softmin': lambda params: nn.Softmin(dim=params.get('dim', -1)),
        'Softplus': lambda params: nn.Softplus(
            beta=params.get('beta', 1),
            threshold=params.get('threshold', 20)
        ),
        'Softshrink': lambda params: nn.Softshrink(lambd=params.get('lambd', 0.5)),
        'Softsign': lambda params: nn.Softsign(),
        'Tanh': lambda params: nn.Tanh(),
        'Tanhshrink': lambda params: nn.Tanhshrink(),
        'Threshold': lambda params: nn.Threshold(
            threshold=params.get('threshold', 0.0),
            value=params.get('value', 0.0),
            inplace=params.get('inplace', False)
        ),
        'None': lambda params: nn.Identity(),
    }

    # Parse activation parameters
    try:
        default_activation_params = ast.literal_eval(activation_params)
    except:
        default_activation_params = {}

    def calculate_conv_output_shape(self, input_shape, kernel_size, stride, padding, dilation=1):
        """Calculate output shape after convolution"""
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        if isinstance(dilation, int): dilation = (dilation, dilation)

        H = ((input_shape[2] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1)
        W = ((input_shape[3] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1)

        return H, W

    def calculate_pool_output_shape(self, input_shape, kernel_size, stride, padding=0):
        """Calculate output shape after pooling"""
        if isinstance(kernel_size, int): kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        
        H = ((input_shape[2] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1)
        W = ((input_shape[3] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1)
        return H, W

    def compile_model(self, mode, LAYER_STACK, activation_function, normalization, 
                    padding_mode, weight_init, activation_params, hyperparameters=None):
        try:
            import torch
            import torch.nn as nn
            import math
            import torch.nn.functional as F
            import ast
            import numpy as np

            # Add inference mode and grad context managers
            with torch.inference_mode(False), torch.set_grad_enabled(True):
                # Move to GPU if available
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




                def _init_layer_parameters(self, layer, layer_def, default_weight_init):
                    """Initialize parameters for different layer types"""
                    # Get weight initialization parameters
                    weight_init = layer_def.get('weight_init', default_weight_init)
                    weight_init_gain = layer_def.get('weight_init_gain', 1.0)
                    weight_init_mode = layer_def.get('weight_init_mode', 'fan_in')
                    weight_init_nonlinearity = layer_def.get('weight_init_nonlinearity', 'relu')
                    
                    if hasattr(layer, 'weight'):
                        if weight_init == 'xavier_uniform':
                            nn.init.xavier_uniform_(layer.weight, gain=weight_init_gain)
                        elif weight_init == 'xavier_normal':
                            nn.init.xavier_normal_(layer.weight, gain=weight_init_gain)
                        elif weight_init == 'kaiming_uniform':
                            nn.init.kaiming_uniform_(layer.weight, mode=weight_init_mode, 
                                                   nonlinearity=weight_init_nonlinearity)
                        elif weight_init == 'kaiming_normal':
                            nn.init.kaiming_normal_(layer.weight, mode=weight_init_mode, 
                                                  nonlinearity=weight_init_nonlinearity)
                        elif weight_init == 'orthogonal':
                            nn.init.orthogonal_(layer.weight, gain=weight_init_gain)

                    # Bias initialization
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        bias_init = layer_def.get('bias_init', 'zeros')
                        bias_init_value = layer_def.get('bias_init_value', 0.0)
                        
                        if bias_init == 'zeros':
                            nn.init.zeros_(layer.bias)
                        elif bias_init == 'ones':
                            nn.init.ones_(layer.bias)
                        elif bias_init == 'normal':
                            nn.init.normal_(layer.bias)
                        elif bias_init == 'uniform':
                            nn.init.uniform_(layer.bias)
                        elif bias_init == 'constant':
                            nn.init.constant_(layer.bias, bias_init_value)

                # Find input shape
                input_shape = None
                for layer in LAYER_STACK:
                    if layer['type'] == 'Input':
                        input_shape = layer['input_shape']
                        break
                
                if input_shape is None:
                    raise ValueError("Input shape not specified. Please add an NntInputLayer.")

                # Create empty tensor to track shape
                current_shape = [1] + input_shape  # Add batch dimension
                layers = []

                # Generate script
                script = self.generate_model_script(LAYER_STACK)
                if mode == "Only create script":
                    return (None, "Script generated successfully", script)

                # Process each layer
                for layer_def in LAYER_STACK:
                    layer_type = layer_def['type']
                    print(f"\nProcessing layer: {layer_type}")
                    print(f"Input shape: {current_shape}")

                    if layer_type == 'Input':
                        continue

                    elif layer_type in ['Conv1d', 'Conv2d', 'Conv3d']:
                        # Get input channels from current tensor
                        in_channels = current_shape[1]
                        
                        # Create conv layer with all parameters
                        conv_class = getattr(nn, layer_type)
                        conv_params = {
                            'in_channels': in_channels,
                            'out_channels': layer_def['out_channels'],
                            'kernel_size': layer_def['kernel_size'],
                            'stride': layer_def['stride'],
                            'padding': layer_def['padding'],
                            'padding_mode': layer_def['padding_mode'],
                            'dilation': layer_def['dilation'],
                            'groups': layer_def['groups'],
                            'bias': layer_def.get('use_bias', True)
                        }
                        
                        conv_layer = conv_class(**conv_params).to(device)
                        
                        # Initialize weights with full parameter set
                        _init_layer_parameters(self, conv_layer, layer_def, weight_init)

                        layers.append(conv_layer)
                        
                        # Update shape after conv
                        H, W = self.calculate_conv_output_shape(
                            input_shape=current_shape,
                            kernel_size=layer_def['kernel_size'],
                            stride=layer_def['stride'],
                            padding=layer_def['padding'],
                            dilation=layer_def['dilation']
                        )
                        current_shape = [current_shape[0], layer_def['out_channels'], H, W]

                        # Add normalization if specified
                        if layer_def.get('normalization', 'None') != 'None':
                            norm_type = layer_def['normalization']
                            if norm_type == 'BatchNorm':
                                norm_type = f'BatchNorm{layer_type[-2]}d'
                            
                            norm_class = getattr(nn, norm_type)
                            norm_layer = norm_class(
                                layer_def['out_channels'],
                                eps=layer_def.get('norm_eps', 1e-5),
                                momentum=layer_def.get('norm_momentum', 0.1),
                                affine=layer_def.get('norm_affine', True),
                                track_running_stats=layer_def.get('track_running_stats', True)
                            ).to(device)
                            layers.append(norm_layer)

                        # Add activation with parameters
                        if layer_def.get('activation', 'None') != 'None':
                            activation_class = getattr(nn, layer_def['activation'])
                            activation_params = {
                                'inplace': layer_def.get('inplace', False)
                            }
                            
                            # Check if the activation function supports the 'negative_slope' argument
                            if 'negative_slope' in inspect.signature(activation_class).parameters:
                                activation_params['negative_slope'] = layer_def.get('alpha', 0.01)
                            
                            activation = activation_class(**activation_params).to(device)
                            layers.append(activation)

                        # Add dropout if specified
                        if layer_def.get('dropout_rate', 0) > 0:
                            dropout = nn.Dropout(p=layer_def['dropout_rate']).to(device)
                            layers.append(dropout)

                    elif layer_type == 'Linear':
                        # Add flattening if needed
                        if len(current_shape) > 2:
                            flatten = nn.Flatten().to(device)
                            layers.append(flatten)
                            current_shape = [current_shape[0], np.prod(current_shape[1:])]

                        linear = nn.Linear(
                            in_features=current_shape[1],
                            out_features=layer_def['num_nodes'],
                            bias=layer_def.get('use_bias', True)
                        ).to(device)

                        # Initialize weights with full parameter set
                        _init_layer_parameters(self, linear, layer_def, weight_init)
                        
                        layers.append(linear)
                        current_shape = [current_shape[0], layer_def['num_nodes']]

                        # Add activation if specified
                        if layer_def.get('activation', 'None') != 'None':
                            activation_class = getattr(nn, layer_def['activation'])
                            activation_params = {
                                'inplace': layer_def.get('inplace', False)
                            }
                            
                            # Check if the activation function supports the 'negative_slope' argument
                            if 'negative_slope' in inspect.signature(activation_class).parameters:
                                activation_params['negative_slope'] = layer_def.get('alpha', 0.01)
                            
                            activation = activation_class(**activation_params).to(device)
                            layers.append(activation)

                        # Add dropout if specified
                        if layer_def.get('dropout_rate', 0) > 0:
                            dropout = nn.Dropout(p=layer_def['dropout_rate']).to(device)
                            layers.append(dropout)

                    elif layer_type == 'Flatten':
                        flatten = nn.Flatten(
                            start_dim=layer_def.get('start_dim', 1),
                            end_dim=layer_def.get('end_dim', -1)
                        ).to(device)
                        layers.append(flatten)
                        current_shape = [current_shape[0], np.prod(current_shape[1:])]

                    elif layer_type == 'Reshape':
                        # Create a custom reshape module
                        class ReshapeLayer(nn.Module):
                            def __init__(self, target_shape):
                                super().__init__()
                                self.target_shape = target_shape

                            def forward(self, x):
                                batch_size = x.size(0)
                                return x.view(batch_size, *self.target_shape)

                        reshape_layer = ReshapeLayer(layer_def['target_shape']).to(device)
                        layers.append(reshape_layer)
                        current_shape = [current_shape[0]] + layer_def['target_shape']

                    elif layer_type in ['MaxPool2d', 'AvgPool2d']:
                        pool_class = getattr(nn, layer_type)
                        pool_params = {
                            'kernel_size': layer_def['kernel_size'],
                            'stride': layer_def.get('stride', None),
                            'padding': layer_def.get('padding', 0),
                            'ceil_mode': layer_def.get('ceil_mode', False)
                        }
                        
                        # Add MaxPool specific parameters
                        if 'MaxPool' in layer_type:
                            pool_params.update({
                                'dilation': layer_def.get('dilation', 1),
                                'return_indices': layer_def.get('return_indices', False)
                            })
                        elif 'AvgPool' in layer_type:
                            pool_params['count_include_pad'] = layer_def.get('count_include_pad', True)
                            
                        pool_layer = pool_class(**pool_params).to(device)
                        layers.append(pool_layer)
                        
                        # Update shape after pooling
                        H, W = self.calculate_pool_output_shape(
                            current_shape,
                            layer_def['kernel_size'],
                            layer_def.get('stride', layer_def['kernel_size']),
                            layer_def.get('padding', 0)
                        )
                        current_shape = [current_shape[0], current_shape[1], H, W]

                        # Add flattening if specified
                        if layer_def.get('flatten_output', False):
                            flatten = nn.Flatten().to(device)
                            layers.append(flatten)
                            current_shape = [current_shape[0], np.prod(current_shape[1:])]

                    elif layer_type in ['BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'LayerNorm']:
                        norm_params = {
                            'eps': layer_def.get('norm_eps', 1e-5),
                            'momentum': layer_def.get('norm_momentum', 0.1),
                            'affine': layer_def.get('norm_affine', True),
                            'track_running_stats': layer_def.get('track_running_stats', True)
                        }
                        
                        if layer_type == 'LayerNorm':
                            norm_params['normalized_shape'] = current_shape[1:]
                        else:
                            norm_params['num_features'] = current_shape[1]
                            
                        norm_class = getattr(nn, layer_type)
                        norm_layer = norm_class(**norm_params).to(device)
                        layers.append(norm_layer)

                # Create the model and move to device
                model = nn.Sequential(*layers).to(device)
                
                # Set requires_grad for all parameters
                for param in model.parameters():
                    param.requires_grad_(True)
                
                # Store model attributes
                model.input_shape = input_shape
                model.device = device
                if hyperparameters is not None:
                    model.hyperparameters = hyperparameters

                # Ensure model is in training mode
                model.train()
                
                return (model, "Model compiled successfully", script)

        except Exception as e:
            import traceback
            error_msg = f"Error compiling model: {str(e)}\n{traceback.format_exc()}"
            return (None, error_msg, "")

    def generate_model_script(self, LAYER_STACK):
        """Helper method to generate complete model script"""
        script_lines = [
            "import torch",
            "import torch.nn as nn",
            "import math",
            "",
            "class GeneratedModel(nn.Sequential):",
            "    def __init__(self):",
            "        # Initialize layers"
        ]
        
        # Track current shape and layer numbers
        input_shape = None
        layer_idx = 0
        current_shape = None
        
        for layer in LAYER_STACK:
            layer_type = layer.get('type', 'Unknown')
            
            if layer_type == 'Input':
                input_shape = layer.get('input_shape', [1, 28, 28])
                current_shape = [1] + input_shape  # Add batch dimension
                script_lines.append(f"        self.input_shape = {input_shape}")
                script_lines.append("")
                continue
                
            # Generate layer definition based on type
            if layer_type in ['Conv1d', 'Conv2d', 'Conv3d']:
                script_lines.append(f"        self.conv{layer_idx} = nn.{layer_type}(")
                script_lines.append(f"            in_channels={current_shape[1]},")
                script_lines.append(f"            out_channels={layer['out_channels']},")
                script_lines.append(f"            kernel_size={layer['kernel_size']},")
                script_lines.append(f"            stride={layer['stride']},")
                script_lines.append(f"            padding={layer['padding']},")
                script_lines.append(f"            bias={layer.get('use_bias', True)}")
                script_lines.append("        )")
                script_lines.append("")
                
                # Update shape
                H = (current_shape[2] + 2*layer['padding'] - layer['kernel_size']) // layer['stride'] + 1
                W = (current_shape[3] + 2*layer['padding'] - layer['kernel_size']) // layer['stride'] + 1
                current_shape = [current_shape[0], layer['out_channels'], H, W]
                
            elif layer_type == 'Linear':
                script_lines.append(f"        self.linear{layer_idx} = nn.Linear(")
                script_lines.append(f"            in_features={current_shape[1]},")
                script_lines.append(f"            out_features={layer['num_nodes']},")
                script_lines.append(f"            bias={layer.get('use_bias', True)}")
                script_lines.append("        )")
                
                if layer.get('activation', 'None') != 'None':
                    if layer['activation'] == 'Softmax':
                        script_lines.append(f"        self.activation{layer_idx} = nn.{layer['activation']}(dim=1)")
                    else:
                        script_lines.append(f"        self.activation{layer_idx} = nn.{layer['activation']}()")
                script_lines.append("")
                
                current_shape = [current_shape[0], layer['num_nodes']]
                
            elif layer_type == 'Flatten':
                script_lines.append(f"        self.flatten{layer_idx} = nn.Flatten()")
                script_lines.append("")
                current_shape = [current_shape[0], np.prod(current_shape[1:])]

            elif layer_type == 'Reshape':
                script_lines.append(f"        class Reshape{layer_idx}(nn.Module):")
                script_lines.append("            def __init__(self):")
                script_lines.append("                super().__init__()")
                script_lines.append(f"                self.target_shape = {layer['target_shape']}")
                script_lines.append("")
                script_lines.append("            def forward(self, x):")
                script_lines.append("                batch_size = x.size(0)")
                script_lines.append("                return x.view(batch_size, *self.target_shape)")
                script_lines.append("")
                script_lines.append(f"        self.reshape{layer_idx} = Reshape{layer_idx}()")
                script_lines.append("")

                # Update current shape
                current_shape = [current_shape[0]] + layer['target_shape'] 

            elif layer_type in ['MaxPool2d', 'AvgPool2d']:
                script_lines.append(f"        self.pool{layer_idx} = nn.{layer_type}(")
                script_lines.append(f"            kernel_size={layer['kernel_size']},")
                script_lines.append(f"            stride={layer.get('stride', layer['kernel_size'])},")
                script_lines.append(f"            padding={layer.get('padding', 0)}")
                script_lines.append("        )")
                script_lines.append("")
                
                # Update shape
                H = (current_shape[2] - layer['kernel_size']) // layer.get('stride', layer['kernel_size']) + 1
                W = (current_shape[3] - layer['kernel_size']) // layer.get('stride', layer['kernel_size']) + 1
                current_shape = [current_shape[0], current_shape[1], H, W]

            elif layer_type in ['BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'LayerNorm']:
                script_lines.append(f"        self.norm{layer_idx} = nn.{layer_type}(")
                
                # For LayerNorm
                if layer_type == 'LayerNorm':
                    script_lines.append(f"            normalized_shape={current_shape[1]},")
                else:
                    # For BatchNorm variants
                    script_lines.append(f"            num_features={current_shape[1]},")
                
                script_lines.append(f"            eps={layer.get('eps', 1e-5)},")
                script_lines.append(f"            momentum={layer.get('momentum', 0.1)},")
                script_lines.append(f"            affine={layer.get('affine', True)}")
                script_lines.append("        )")
                script_lines.append("")


            # Add batch norm if specified
            if layer.get('normalization', 'None') != 'None':
                    norm_type = layer['normalization']
                    if norm_type == 'BatchNorm':
                        if layer_type.endswith('1d'): norm_type = 'BatchNorm1d'
                        elif layer_type.endswith('2d'): norm_type = 'BatchNorm2d'
                        elif layer_type.endswith('3d'): norm_type = 'BatchNorm3d'
                        
                    # Get the number of features for normalization
                    num_features = None
                    if 'out_channels' in layer:
                        num_features = layer['out_channels']
                    elif 'num_nodes' in layer:
                        num_features = layer['num_nodes']
                    elif 'num_features' in layer:
                        num_features = layer['num_features']
                    else:
                        # Default to previous layer's output size
                        for prev_layer in reversed(LAYER_STACK[:layer_idx]):
                            if 'out_channels' in prev_layer:
                                num_features = prev_layer['out_channels']
                                break
                            elif 'num_nodes' in prev_layer:
                                num_features = prev_layer['num_nodes']
                                break

                    if num_features is not None:
                        script_lines.append(f"        self.norm{layer_idx} = nn.{norm_type}({num_features})")
                        script_lines.append("")
                
            layer_idx += 1
        
        # Build layers list for Sequential
        layers_list = []
        layer_idx = 0
        for layer in LAYER_STACK:
            if layer['type'] == 'Input':
                continue
                
            if layer['type'] in ['Conv1d', 'Conv2d', 'Conv3d']:
                layers_list.extend([
                    f"self.conv{layer_idx}",
                    f"self.norm{layer_idx}" if layer.get('normalization', 'None') != 'None' else None,
                    f"self.activation{layer_idx}" if layer.get('activation', 'None') != 'None' else None
                ])
            elif layer['type'] == 'Linear':
                layers_list.extend([
                    f"self.linear{layer_idx}",
                    f"self.activation{layer_idx}" if layer.get('activation', 'None') != 'None' else None
                ])
            elif layer['type'] == 'Reshape':
                layers_list.append(f"self.reshape{layer_idx}")
            elif layer['type'] == 'Flatten':
                layers_list.append(f"self.flatten{layer_idx}")
            elif layer['type'] in ['MaxPool2d', 'AvgPool2d']:
                layers_list.append(f"self.pool{layer_idx}")
            elif layer['type'] in ['BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 'LayerNorm']:
                layers_list.append(f"self.norm{layer_idx}")    
            layer_idx += 1
        
        # Filter out None values and create the sequential constructor
        layers_list = [l for l in layers_list if l is not None]
        if layers_list:
            script_lines.append("        # Combine layers")
            script_lines.append("        super().__init__(")
            for layer in layers_list:
                script_lines.append(f"            {layer},")
            script_lines.append("        )")
            script_lines.append("")
        
        # Add example usage
        script_lines.extend([
            "# Example usage:",
            "# -----------------------------",
            "# model = GeneratedModel()",
            f"# input_tensor = torch.randn(1, {', '.join(str(x) for x in input_shape)})  # Example input tensor",
            "# output = model(input_tensor)",
            "# print(f'Input shape: {list(input_tensor.shape)}')",
            "# print(f'Output shape: {list(output.shape)}')"
        ])
        
        return "\n".join(script_lines)

    def create_pooling_layer(self, pool_type, params):
        """Helper method to create various types of pooling layers"""
        import torch.nn as nn

        if pool_type.endswith(('Pool1d', 'Pool2d', 'Pool3d')):
            pool_class = getattr(nn, pool_type)
            
            if pool_type.startswith('Adaptive'):
                return pool_class(
                    output_size=params['output_size'],
                    return_indices=params.get('return_indices', False)
                )
            elif pool_type.startswith('MaxUnpool'):
                return pool_class(
                    kernel_size=params['kernel_size'],
                    stride=params['stride'],
                    padding=params['padding']
                )
            elif pool_type.startswith('Fractional'):
                return pool_class(
                    kernel_size=params['kernel_size'],
                    output_ratio=1/params['fractional_factor'],
                    return_indices=params.get('return_indices', False)
                )
            elif pool_type.startswith('LP'):
                return pool_class(
                    norm_type=int(params['norm_type']),
                    kernel_size=params['kernel_size'],
                    stride=params.get('stride', None),
                    ceil_mode=params.get('ceil_mode', False)
                )
            else:  # Standard Max/Avg Pool
                kwargs = {
                    'kernel_size': params['kernel_size'],
                    'stride': params['stride'],
                    'padding': params['padding'],
                    'ceil_mode': params.get('ceil_mode', False)
                }
                
                if 'MaxPool' in pool_type:
                    kwargs.update({
                        'dilation': params.get('dilation', 1),
                        'return_indices': params.get('return_indices', False)
                    })
                elif 'AvgPool' in pool_type:
                    kwargs['count_include_pad'] = params.get('count_include_pad', True)
                    
                return pool_class(**kwargs)
        return None

    def adjust_linear_in_features(self, model, input_shape):
        import torch
        import torch.nn as nn
        dummy_input = torch.zeros(1, *input_shape)
        modules = list(model.children())
        input = dummy_input
        new_modules = []
        for idx, module in enumerate(modules):
            if isinstance(module, nn.Linear):
                in_features = input.nelement() // input.size(0)
                out_features = module.out_features
                bias = module.bias is not None
                weight_init_method = getattr(module, 'weight_init_method', 'default')
                bias_init_method = getattr(module, 'bias_init_method', 'default')
                linear_layer = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
                self.apply_weight_init(linear_layer.weight, weight_init_method)
                if linear_layer.bias is not None:
                    self.apply_bias_init(linear_layer.bias, bias_init_method)
                new_modules.append(linear_layer)
                module = linear_layer
                input = module(input)
            else:
                new_modules.append(module)
                input = module(input)
        model = nn.Sequential(*new_modules)
        return model

    def apply_weight_init(self, weight, method):
        import torch.nn.init as init
        if method == 'normal':
            init.normal_(weight)
        elif method == 'uniform':
            init.uniform_(weight)
        elif method == 'xavier_normal':
            init.xavier_normal_(weight)
        elif method == 'xavier_uniform':
            init.xavier_uniform_(weight)
        elif method == 'kaiming_normal':
            init.kaiming_normal_(weight)
        elif method == 'kaiming_uniform':
            init.kaiming_uniform_(weight)
        elif method == 'orthogonal':
            init.orthogonal_(weight)

    def apply_bias_init(self, bias, method):
        import torch.nn.init as init
        if method == 'zeros':
            init.constant_(bias, 0)
        elif method == 'ones':
            init.constant_(bias, 1)
        elif method == 'normal':
            init.normal_(bias)
        elif method == 'uniform':
            init.uniform_(bias)

    def _create_vanilla_attention(self, layer_def):
        class VanillaAttention(nn.Module):
            def __init__(self, embed_dim, attention_type='scaled_dot_product', dropout=0.0):
                super().__init__()
                self.attention_type = attention_type
                self.scale = embed_dim ** -0.5 if attention_type == 'scaled_dot_product' else 1.0
                self.q_proj = nn.Linear(embed_dim, embed_dim)
                self.k_proj = nn.Linear(embed_dim, embed_dim)
                self.v_proj = nn.Linear(embed_dim, embed_dim)
                self.dropout = nn.Dropout(dropout)

                if attention_type == 'additive':
                    self.energy_proj = nn.Linear(embed_dim, 1)

            def forward(self, query, key, value):
                q = self.q_proj(query)
                k = self.k_proj(key)
                v = self.v_proj(value)

                if self.attention_type in ['dot_product', 'scaled_dot_product']:
                    attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
                elif self.attention_type == 'additive':
                    q_expanded = q.unsqueeze(-2).expand(-1, -1, k.size(-2), -1)
                    k_expanded = k.unsqueeze(-3).expand(-1, q.size(-2), -1, -1)
                    combined = torch.tanh(q_expanded + k_expanded)
                    attn = self.energy_proj(combined).squeeze(-1)

                attn = F.softmax(attn, dim=-1)
                attn = self.dropout(attn)
                output = torch.matmul(attn, v)
                return output, attn

        return VanillaAttention(
            embed_dim=layer_def['embed_dim'],
            attention_type=layer_def['attention_type'],
            dropout=layer_def['dropout']
        )

    def _create_linear_attention(self, layer_def):
        class LinearAttention(nn.Module):
            def __init__(self, embed_dim, num_heads, feature_map='elu', eps=1e-6):
                super().__init__()
                self.embed_dim = embed_dim
                self.num_heads = num_heads
                self.head_dim = embed_dim // num_heads
                self.feature_map_fn = {
                    'elu': lambda x: F.elu(x) + 1,
                    'relu': F.relu,
                    'softmax': lambda x: F.softmax(x, dim=-1)
                }[feature_map]
                self.eps = eps
                
                self.q_proj = nn.Linear(embed_dim, embed_dim)
                self.k_proj = nn.Linear(embed_dim, embed_dim)
                self.v_proj = nn.Linear(embed_dim, embed_dim)
                self.out_proj = nn.Linear(embed_dim, embed_dim)

            def forward(self, x):
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)

                q = self.feature_map_fn(q)
                k = self.feature_map_fn(k)

                kv = torch.matmul(k.transpose(-2, -1), v)
                z = 1 / (torch.matmul(q, k.sum(dim=-2).unsqueeze(-1)) + self.eps)
                output = torch.matmul(q, kv) * z
                
                return self.out_proj(output)

        return LinearAttention(
            embed_dim=layer_def['embed_dim'],
            num_heads=layer_def['num_heads'],
            feature_map=layer_def['feature_map'],
            eps=layer_def.get('eps', 1e-6)
        )

    def _create_transformer_xl_attention(self, layer_def):
        class TransformerXLAttention(nn.Module):
            def __init__(self, d_model, num_heads, mem_len=0):
                super().__init__()
                self.d_model = d_model
                self.num_heads = num_heads
                self.mem_len = mem_len
                self.head_dim = d_model // num_heads

                self.q_proj = nn.Linear(d_model, d_model)
                self.k_proj = nn.Linear(d_model, d_model)
                self.v_proj = nn.Linear(d_model, d_model)
                self.o_proj = nn.Linear(d_model, d_model)
                
                self.r_proj = nn.Linear(d_model, d_model)
                self.r_emb = nn.Parameter(torch.Tensor(mem_len, self.head_dim))
                nn.init.normal_(self.r_emb, mean=0.0, std=0.02)

            def forward(self, x, memory=None):
                B, L, _ = x.size()
                
                q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
                k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
                v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)

                if memory is not None:
                    k = torch.cat([memory, k], dim=1)
                    v = torch.cat([memory, v], dim=1)

                attn = torch.matmul(q, k.transpose(-2, -1))
                
                if self.mem_len > 0:
                    r = self.r_proj(self.r_emb[:L])
                    r_term = torch.matmul(q, r.transpose(-2, -1))
                    attn = attn + r_term

                attn = F.softmax(attn / math.sqrt(self.head_dim), dim=-1)
                output = torch.matmul(attn, v)
                
                return self.o_proj(output.view(B, L, -1))

        return TransformerXLAttention(
            d_model=layer_def['d_model'],
            num_heads=layer_def['num_heads'],
            mem_len=layer_def['mem_len']
        )

    def _create_reformer_attention(self, layer_def):
        class ReformerAttention(nn.Module):
            def __init__(self, embed_dim, num_heads, num_buckets=32, bucket_size=64, num_hashes=8):
                super().__init__()
                self.embed_dim = embed_dim
                self.num_heads = num_heads
                self.num_buckets = num_buckets
                self.bucket_size = bucket_size
                self.num_hashes = num_hashes
                
                self.q_proj = nn.Linear(embed_dim, embed_dim)
                self.k_proj = nn.Linear(embed_dim, embed_dim)
                self.v_proj = nn.Linear(embed_dim, embed_dim)
                self.out_proj = nn.Linear(embed_dim, embed_dim)

            def forward(self, x):
                # Simplified Reformer attention - in practice would use LSH
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
                
                attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim)
                attn = F.softmax(attn, dim=-1)
                output = torch.matmul(attn, v)
                
                return self.out_proj(output)

        return ReformerAttention(
            embed_dim=layer_def['embed_dim'],
            num_heads=layer_def['num_heads'],
            num_buckets=layer_def['num_buckets'],
            bucket_size=layer_def['bucket_size'],
            num_hashes=layer_def['num_hashes']
        )

    def _create_local_attention(self, layer_def):
        class LocalAttention(nn.Module):
            def __init__(self, embed_dim, num_heads, window_size, look_behind=0, look_ahead=0):
                super().__init__()
                self.embed_dim = embed_dim
                self.num_heads = num_heads
                self.window_size = window_size
                self.look_behind = look_behind
                self.look_ahead = look_ahead
                
                self.q_proj = nn.Linear(embed_dim, embed_dim)
                self.k_proj = nn.Linear(embed_dim, embed_dim)
                self.v_proj = nn.Linear(embed_dim, embed_dim)
                self.out_proj = nn.Linear(embed_dim, embed_dim)

            def create_local_attention_mask(self, size):
                mask = torch.ones(size, size, dtype=torch.bool)
                for i in range(size):
                    start = max(0, i - self.look_behind)
                    end = min(size, i + self.look_ahead + 1)
                    mask[i, start:end] = False
                return mask

            def forward(self, x):
                B, L, _ = x.size()
                
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)

                mask = self.create_local_attention_mask(L)
                
                attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim)
                attn = attn.masked_fill(mask, float('-inf'))
                attn = F.softmax(attn, dim=-1)
                
                output = torch.matmul(attn, v)
                return self.out_proj(output)

        return LocalAttention(
            embed_dim=layer_def['embed_dim'],
            num_heads=layer_def['num_heads'],
            window_size=layer_def['window_size'],
            look_behind=layer_def['look_behind'],
            look_ahead=layer_def['look_ahead']
        )

    def _create_rotary_embedding(self, layer_def):
        class RotaryEmbedding(nn.Module):
            def __init__(self, dim, max_freq=10, base=10000):
                super().__init__()
                inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
                self.register_buffer('inv_freq', inv_freq)
                self.max_seq_len_cached = None
                self.cos_cached = None
                self.sin_cached = None

            def forward(self, x, seq_len=None):
                if seq_len != self.max_seq_len_cached:
                    self.max_seq_len_cached = seq_len if seq_len is not None else x.shape[1]
                    t = torch.arange(self.max_seq_len_cached, device=x.device).type_as(self.inv_freq)
                    freqs = torch.einsum('i,j->ij', t, self.inv_freq)
                    emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0)
                    self.cos_cached = emb.cos()
                    self.sin_cached = emb.sin()
                return x * self.cos_cached + self._rotate_half(x) * self.sin_cached

            def _rotate_half(self, x):
                x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
                return torch.cat((-x2, x1), dim=-1)

        return RotaryEmbedding(
            dim=layer_def['dim'],
            max_freq=layer_def['max_freq'],
            base=layer_def['base']
        )

    def _create_relative_position_bias(self, layer_def):
        class RelativePositionBias(nn.Module):
            def __init__(self, num_buckets, max_distance, num_heads, causal=False):
                super().__init__()
                self.num_buckets = num_buckets
                self.max_distance = max_distance
                self.num_heads = num_heads
                self.causal = causal
                
                self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

            def _relative_position_bucket(self, relative_position):
                ret = 0
                n = -relative_position
                if self.causal:
                    if ret < self.num_buckets:
                        n = torch.clamp(n, 0, self.max_distance)
                    ret += self.num_buckets
                else:
                    n = torch.abs(n)
                    n = torch.clamp(n, 0, self.max_distance)
                return ret

            def forward(self, qlen, klen):
                context_position = torch.arange(qlen, dtype=torch.long)[:, None]
                memory_position = torch.arange(klen, dtype=torch.long)[None, :]
                relative_position = memory_position - context_position
                rp_bucket = self._relative_position_bucket(relative_position)
                values = self.relative_attention_bias(rp_bucket)
                return values.permute(2, 0, 1).unsqueeze(0)

        return RelativePositionBias(
            num_buckets=layer_def['num_buckets'],
            max_distance=layer_def['max_distance'],
            num_heads=layer_def['num_heads'],
            causal=layer_def['causal']
        )

    def _init_layer_parameters(self, layer, layer_def, default_weight_init):
        # Get weight initialization parameters
        weight_init = layer_def.get('weight_init', default_weight_init)  # Use passed default
        weight_init_gain = layer_def.get('weight_init_gain', 1.0)
        weight_init_mode = layer_def.get('weight_init_mode', 'fan_in')
        weight_init_nonlinearity = layer_def.get('weight_init_nonlinearity', 'relu')

        """Initialize parameters for different layer types"""
        if hasattr(layer, 'weight'):
            if weight_init == 'xavier_uniform':
                nn.init.xavier_uniform_(layer.weight, 
                    gain=nn.init.calculate_gain(activation_function.lower()))
            elif weight_init == 'xavier_normal':
                nn.init.xavier_normal_(layer.weight,
                    gain=nn.init.calculate_gain(activation_function.lower()))
            elif weight_init == 'kaiming_uniform':
                nn.init.kaiming_uniform_(layer.weight,
                    nonlinearity=activation_function.lower())
            elif weight_init == 'kaiming_normal':
                nn.init.kaiming_normal_(layer.weight,
                    nonlinearity=activation_function.lower())
            elif weight_init == 'orthogonal':
                nn.init.orthogonal_(layer.weight,
                    gain=nn.init.calculate_gain(activation_function.lower()))

        if hasattr(layer, 'bias') and layer.bias is not None:
            nn.init.zeros_(layer.bias)

    def _create_alibi_bias(self, layer_def):
        class AliBi(nn.Module):
            def __init__(self, num_heads, max_seq_length, causal=False):
                super().__init__()
                self.num_heads = num_heads
                self.max_seq_length = max_seq_length
                self.causal = causal
                
                slopes = torch.Tensor(self._get_slopes(num_heads))
                alibi = self._get_alibi_mask(max_seq_length, slopes)
                self.register_buffer('alibi', alibi)

            def _get_slopes(self, n):
                def get_slopes_power_of_2(n):
                    start = 2**(-10)  # 2^(-2-8) = 2^(-10)
                    ratio = 2**(-8)
                    return [start * ratio ** i for i in range(n)]

                if math.log2(n).is_integer():
                    return get_slopes_power_of_2(n)
                else:
                    closest_power_of_2 = 2 ** math.floor(math.log2(n))
                    return get_slopes_power_of_2(closest_power_of_2) + \
                        get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

            def _get_alibi_mask(self, seq_length, slopes):
                alibi = slopes.unsqueeze(1).unsqueeze(1) * \
                        torch.arange(seq_length).unsqueeze(0).unsqueeze(0).expand(
                            self.num_heads, -1, -1)
                if self.causal:
                    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
                    alibi.masked_fill_(mask, float('-inf'))
                return alibi

            def forward(self, attention_scores):
                return attention_scores + self.alibi[:, :attention_scores.shape[-2], :attention_scores.shape[-1]]

        return AliBi(
            num_heads=layer_def['num_heads'],
            max_seq_length=layer_def['max_seq_length'],
            causal=layer_def['causal']
        )

class NntLoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": ("STRING", {"default": "model.pth"}),
                "directory": ("STRING", {"default": ""}),
                "load_format": (["PyTorch Model", "State Dict", "TorchScript", "ONNX", "TorchScript Mobile", "Quantized", "SafeTensors"], 
                    {"default": "PyTorch Model"}),
                "load_optimizer": (["True", "False"], {"default": "False"}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "load_model"
    OUTPUT_NODE = True
    CATEGORY = "NNT Neural Network Toolkit/Models"

    @staticmethod
    def get_default_model_path():
        """Get default path for model storage"""
        import os
        import folder_paths

        if not "nnt_models" in folder_paths.folder_names_and_paths:
            model_path = os.path.join(folder_paths.models_dir, "nnt_models")
            folder_paths.add_model_folder_path("nnt_models", model_path)
            return model_path
        
        return os.path.join(folder_paths.models_dir, "nnt_models")

    def load_model(self, filename, directory, load_format, load_optimizer):
        import torch
        from pathlib import Path
        
        try:
            directory = directory if directory.strip() else self.get_default_model_path()
            file_path = Path(directory) / filename
            base_path = file_path.with_suffix('')
            
            if not file_path.exists() and not base_path.with_suffix('.pt').exists() and \
                not base_path.with_suffix('.onnx').exists() and not base_path.with_suffix('.ptl').exists() and \
                not base_path.with_suffix('.quantized.pth').exists():
                return (None, f"Error: No model file found at {file_path}")

            if load_format == "SafeTensors":
                from safetensors.torch import load_file
                from torch import nn
                safetensor_path = base_path.with_suffix('.safetensors')
                state_dict = load_file(str(safetensor_path))
                
                # Create simple Sequential model from state dict structure
                layers = []
                prev_layer = None
                for key, tensor in state_dict.items():
                    if 'weight' in key:
                        layer_name = key.rsplit('.', 1)[0]
                        if 'linear' in key:
                            in_features = tensor.size(1)
                            out_features = tensor.size(0)
                            layers.append(nn.Linear(in_features, out_features))
                        elif 'conv' in key:
                            out_channels = tensor.size(0)
                            in_channels = tensor.size(1)
                            kernel_size = tensor.size(2)
                            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))

                model = nn.Sequential(*layers)
                model.load_state_dict(state_dict)
                status_msg = f"SafeTensors model loaded from {safetensor_path}"
        
            elif load_format == "PyTorch Model":
                loaded_data = torch.load(str(file_path))
                if isinstance(loaded_data, dict):
                    model = loaded_data['model']
                    if load_optimizer == "True" and 'optimizer_state_dict' in loaded_data:
                        optimizer = loaded_data.get('optimizer', 'Adam')
                        optimizer_class = getattr(torch.optim, optimizer)
                        optimizer = optimizer_class(model.parameters())
                        optimizer.load_state_dict(loaded_data['optimizer_state_dict'])
                        status_msg = f"Model and optimizer loaded from {file_path}"
                    else:
                        status_msg = f"Model loaded from {file_path}"
                else:
                    model = loaded_data
                    status_msg = f"Model loaded from {file_path}"

            elif load_format == "State Dict":
                loaded_data = torch.load(str(file_path))
                if not isinstance(loaded_data, dict) or 'model_state_dict' not in loaded_data:
                    return (None, "Invalid state dict file")
                
                from .neural_network_toolkit import NntCompileModel
                model = NntCompileModel().compile_model(loaded_data['model_config'])
                model.load_state_dict(loaded_data['model_state_dict'])
                
                if load_optimizer == "True" and 'optimizer_state_dict' in loaded_data:
                    optimizer = loaded_data.get('optimizer_type', 'Adam')
                    optimizer_class = getattr(torch.optim, optimizer)
                    optimizer = optimizer_class(model.parameters())
                    optimizer.load_state_dict(loaded_data['optimizer_state_dict'])
                    status_msg = f"Model state dict and optimizer loaded from {file_path}"
                else:
                    status_msg = f"Model state dict loaded from {file_path}"

            elif load_format == "TorchScript":
                ts_path = base_path.with_suffix('.pt')
                model = torch.jit.load(str(ts_path))
                status_msg = f"TorchScript model loaded from {ts_path}"

            elif load_format == "ONNX":
                import onnx
                import onnx2pytorch
                onnx_path = base_path.with_suffix('.onnx')
                onnx_model = onnx.load(str(onnx_path))
                model = onnx2pytorch.ConvertModel(onnx_model)
                status_msg = f"ONNX model loaded from {onnx_path}"

            elif load_format == "TorchScript Mobile":
                ptl_path = base_path.with_suffix('.ptl')
                model = torch.jit.load(str(ptl_path))
                status_msg = f"Mobile TorchScript loaded from {ptl_path}"

            elif load_format == "Quantized":
                quantized_path = base_path.with_suffix('.quantized.pth')
                state_dict = torch.load(str(quantized_path))
                
                # Create quantized model structure based on state dict
                from .neural_network_toolkit import NntCompileModel
                model = NntCompileModel().compile_model(state_dict['model_config'])
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                torch.quantization.prepare(model, inplace=True)
                quantized_model = torch.quantization.convert(model, inplace=False)
                quantized_model.load_state_dict(state_dict)
                model = quantized_model
                status_msg = f"Quantized model loaded from {quantized_path}"

            return (model, status_msg)

        except Exception as e:
            return (None, f"Error loading model: {str(e)}")


class NntVisualizeGraph:
   @classmethod
   def INPUT_TYPES(cls):
       return {
           "required": {
               "MODEL": ("MODEL",),
               "input_data": ("TENSOR",),
               "image_width": ("INT", {
                   "default": 1024,
                   "min": 256,
                   "max": 4096,
                   "step": 64
               }),
               "image_height": ("INT", {
                   "default": 768,
                   "min": 256,
                   "max": 4096,
                   "step": 64
               }),
           }
       }

   RETURN_TYPES = ("IMAGE", "STRING")
   FUNCTION = "visualize_graph"
   OUTPUT_NODE = True
   CATEGORY = "NNT Neural Network Toolkit/Models"

   def visualize_graph(self, MODEL, input_data, image_width, image_height):
       try:
           import torch
           from torchviz import make_dot
           import tempfile
           from PIL import Image
           import numpy as np
           import os
           import time

           empty_tensor = torch.zeros(3, image_height, image_width, dtype=torch.uint8)

           #with torch.no_grad():
           output = MODEL(input_data)
           
           timestamp = int(time.time() * 1000)
           temp_dir = tempfile.gettempdir()
           temp_filename = os.path.join(temp_dir, f'model_graph_{timestamp}')

           try:
               dot = make_dot(output, params=dict(list(MODEL.named_parameters())))
               dot.render(temp_filename, format="png", cleanup=True)
               png_path = f"{temp_filename}.png"

               if not os.path.exists(png_path):
                   return (empty_tensor, "Error: Failed to generate graph image")

               img = Image.open(png_path).convert('RGB')
               img = img.resize((image_width, image_height), Image.Resampling.LANCZOS)
               img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)

               try:
                   if os.path.exists(png_path):
                       os.remove(png_path)
                   if os.path.exists(temp_filename):
                       os.remove(temp_filename)
               except:
                   pass

               return (img_tensor, "Model graph generated successfully")

           except Exception as e:
               try:
                   if os.path.exists(png_path):
                       os.remove(png_path)
                   if os.path.exists(temp_filename):
                       os.remove(temp_filename)
               except:
                   pass
               return (empty_tensor, f"Error during graph creation: {str(e)}")

       except Exception as e:
           return (empty_tensor, f"Error generating graph: {str(e)}")

# Add to nnt_models.py

class NntTrainingHyperparameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": TRAINING_PARAMS}

    RETURN_TYPES = ("DICT", "STRING")
    RETURN_NAMES = ("training_params", "config_summary")
    FUNCTION = "configure_training"
    OUTPUT_NODE = True
    CATEGORY = "NNT Neural Network Toolkit/Models"

    def configure_training(self, experiment_name, batch_size, epochs,
                         optimizer, learning_rate, weight_decay, momentum,
                         use_lr_scheduler, scheduler_type, scheduler_step_size,
                         scheduler_gamma, use_early_stopping, patience, min_delta):
        try:
            # Create training parameters dictionary
            training_params = {
                "experiment": {
                    "name": experiment_name,
                },
                "training": {
                    "batch_size": batch_size,
                    "epochs": epochs,
                },
                "optimizer": {
                    "name": optimizer,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "momentum": momentum if optimizer == "SGD" else None
                },
                "lr_scheduler": {
                    "enabled": use_lr_scheduler == "True",
                    "type": scheduler_type,
                    "step_size": scheduler_step_size,
                    "gamma": scheduler_gamma
                },
                "early_stopping": {
                    "enabled": use_early_stopping == "True",
                    "patience": patience,
                    "min_delta": min_delta
                }
            }

            # Create readable summary
            summary = [
                f"Experiment: {experiment_name}",
                "\nTraining:",
                f"- Batch Size: {batch_size}",
                f"- Epochs: {epochs}",
                "\nOptimizer:",
                f"- Type: {optimizer}",
                f"- Learning Rate: {learning_rate}",
                f"- Weight Decay: {weight_decay}",
                f"- Momentum: {momentum if optimizer == 'SGD' else 'N/A'}",
                "\nLR Scheduler:",
                f"- Enabled: {use_lr_scheduler}",
                f"- Type: {scheduler_type if use_lr_scheduler == 'True' else 'N/A'}",
                f"- Step Size: {scheduler_step_size}",
                f"- Gamma: {scheduler_gamma}",
                "\nEarly Stopping:",
                f"- Enabled: {use_early_stopping}",
                f"- Patience: {patience if use_early_stopping == 'True' else 'N/A'}",
                f"- Min Delta: {min_delta}"
            ]

            return (training_params, "\n".join(summary))

        except Exception as e:
            import traceback
            error_msg = f"Error configuring training parameters: {str(e)}\n{traceback.format_exc()}"
            return ({}, error_msg)



class NntVisualizePredictionMetrics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "metrics": ("DICT",),
                "image_width": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 64
                }),
                "image_height": ("INT", {
                    "default": 768,
                    "min": 256,
                    "max": 4096,
                    "step": 64
                }),
                "plot_type": (["confusion_matrix", "class_accuracy", "combined"], {
                    "default": "combined"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "visualize_metrics"
    OUTPUT_NODE = True
    CATEGORY = "NNT Neural Network Toolkit/Models"

    def visualize_metrics(self, metrics, image_width, image_height, plot_type):
        try:
            import torch
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            from PIL import Image
            import io
            
            # Debug print to inspect metrics
            print(f"Received metrics: {metrics}")

            # Check if metrics dictionary is empty or None
            if not metrics:
                raise ValueError("Empty metrics dictionary")

            # Create figure with the right size
            dpi = 100
            fig_width = image_width / dpi
            fig_height = image_height / dpi

            # Check what data is available in metrics
            available_keys = list(metrics.keys())
            print(f"Available metric keys: {available_keys}")

            # Determine metric type and create appropriate visualization
            if isinstance(metrics, dict):
                if 'accuracy' in metrics or 'confusion_matrix' in metrics or 'per_class_accuracy' in metrics:
                    # Classification metrics
                    if plot_type == "combined" or plot_type == "confusion_matrix":
                        fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height), dpi=dpi) if plot_type == "combined" else plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
                        
                        if plot_type == "combined":
                            ax1, ax2 = axes
                        else:
                            ax1 = axes

                        # Plot confusion matrix if available
                        if 'confusion_matrix' in metrics:
                            cm = np.array(metrics['confusion_matrix'])
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
                            ax1.set_title('Confusion Matrix')
                            ax1.set_xlabel('Predicted Class')
                            ax1.set_ylabel('True Class')

                        if plot_type == "combined":
                            # Plot accuracy if available
                            if 'per_class_accuracy' in metrics:
                                accuracies = np.array(metrics['per_class_accuracy'])
                                ax2.bar(range(len(accuracies)), accuracies)
                                ax2.set_title('Per-Class Accuracy')
                                ax2.set_xlabel('Class')
                                ax2.set_ylabel('Accuracy')
                                ax2.set_ylim(0, 1)

                    elif plot_type == "class_accuracy" and 'per_class_accuracy' in metrics:
                        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
                        accuracies = np.array(metrics['per_class_accuracy'])
                        ax.bar(range(len(accuracies)), accuracies)
                        ax.set_title('Per-Class Accuracy')
                        ax.set_xlabel('Class')
                        ax.set_ylabel('Accuracy')
                        ax.set_ylim(0, 1)

                elif 'mse' in metrics or 'mae' in metrics:
                    # Regression metrics
                    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
                    
                    if 'true_values' in metrics and 'predictions' in metrics:
                        true_vals = np.array(metrics['true_values'])
                        pred_vals = np.array(metrics['predictions'])
                        ax.scatter(true_vals, pred_vals, alpha=0.5)
                        
                        # Add perfect prediction line
                        min_val = min(true_vals.min(), pred_vals.min())
                        max_val = max(true_vals.max(), pred_vals.max())
                        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
                        
                        ax.set_title('Predictions vs True Values')
                        ax.set_xlabel('True Values')
                        ax.set_ylabel('Predictions')
                        ax.legend()
                    
                    # Add metrics text
                    txt = []
                    if 'mse' in metrics:
                        txt.append(f'MSE: {metrics["mse"]:.4f}')
                    if 'mae' in metrics:
                        txt.append(f'MAE: {metrics["mae"]:.4f}')
                    if txt:
                        plt.figtext(0.02, 0.98, '\n'.join(txt), va='top')
                else:
                    raise ValueError(f"No supported metrics found in dictionary. Available keys: {available_keys}")
            else:
                raise ValueError(f"Expected dictionary, got {type(metrics)}")

            # Convert plot to tensor
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            
            image = Image.open(buf).convert('RGB')
            result = torch.tensor(np.array(image).astype(np.float32) / 255.0)
            result = torch.unsqueeze(result, 0)
            
            plt.close(fig)
            buf.close()

            # Create summary string
            summary = ["Metrics Visualization Summary:"]
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    summary.append(f"{key}: {value:.4f}")
                elif isinstance(value, list) and len(value) < 10:
                    summary.append(f"{key}: {value}")
                else:
                    summary.append(f"{key}: {type(value)}")

            return (result, "\n".join(summary))

        except Exception as e:
            import traceback
            error_msg = f"Error visualizing metrics: {str(e)}\n{traceback.format_exc()}"
            empty_tensor = torch.zeros((1, image_height, image_width, 3), dtype=torch.float32)
            return (empty_tensor, error_msg)
        
import torch
import numpy as np

class NntInference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "MODEL": ("MODEL",),
                "input_tensor": ("TENSOR",),
                "mode": (["single", "batch", "all"], {
                    "default": "single"
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 9999,
                    "step": 1
                }),
                "batch_size": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 512,
                    "step": 1
                }),
                "output_type": (["raw", "probabilities", "class_predictions"], {
                    "default": "probabilities"
                }),
                "return_confidence": (["True", "False"], {
                    "default": "True"
                }),
            },
            "optional": {
                "device": (["cpu", "cuda"], {"default": "cuda"}),
                "index_list": ("STRING", {
                    "default": "[]",
                    "multiline": False,
                    "placeholder": "e.g., [0,1,4,7] for specific indices"
                }),
                "preprocessing": (["None", "normalize", "standardize"], {
                    "default": "None"
                }),
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "STRING", "DICT")
    RETURN_NAMES = ("output_tensor", "confidence_scores", "inference_info", "metrics")
    FUNCTION = "run_inference"
    CATEGORY = "NNT Neural Network Toolkit/Inference"
    
    def preprocess_input(self, tensor, method):
        if method == "normalize":
            return (tensor - tensor.min()) / (tensor.max() - tensor.min())
        elif method == "standardize":
            mean = tensor.mean()
            std = tensor.std()
            return (tensor - mean) / (std + 1e-8)
        return tensor

    def run_inference(self, MODEL, input_tensor, mode="single", index=0, batch_size=32,
                     output_type="probabilities", return_confidence="True", 
                     device="cuda", index_list="[]", preprocessing="None"):
        try:
            import torch
            import torch.nn.functional as F
            import numpy as np
            import ast

            # Move to specified device
            device = torch.device(device if torch.cuda.is_available() else "cpu")
            MODEL = MODEL.to(device)
            
            # Prepare metrics dictionary
            metrics = {
                "total_samples": 0,
                "processing_time": 0,
                "confidence_stats": {},
                "batch_metrics": []
            }

            # Parse index list if provided
            try:
                specific_indices = ast.literal_eval(index_list) if index_list != "[]" else None
            except:
                specific_indices = None

            # Prepare input based on mode
            if mode == "single":
                selected_input = input_tensor[index:index+1]
            elif mode == "batch":
                if specific_indices:
                    selected_input = input_tensor[specific_indices]
                else:
                    start_idx = index
                    end_idx = min(start_idx + batch_size, len(input_tensor))
                    selected_input = input_tensor[start_idx:end_idx]
            else:  # mode == "all"
                selected_input = input_tensor

            # Preprocess input
            selected_input = self.preprocess_input(selected_input, preprocessing)
            selected_input = selected_input.to(device)

            # Run inference
            MODEL.eval()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            
            with torch.no_grad():
                start_time.record()
                raw_output = MODEL(selected_input)
                end_time.record()
                
                # Process output based on type
                if output_type == "probabilities":
                    output = F.softmax(raw_output, dim=1)
                    confidence_scores = output.max(dim=1)[0]
                elif output_type == "class_predictions":
                    output = raw_output.argmax(dim=1)
                    confidence_scores = F.softmax(raw_output, dim=1).max(dim=1)[0]
                else:  # raw
                    output = raw_output
                    confidence_scores = torch.ones(len(raw_output))

                # Update metrics
                torch.cuda.synchronize()
                metrics["total_samples"] = len(selected_input)
                metrics["processing_time"] = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
                metrics["confidence_stats"] = {
                    "mean": float(confidence_scores.mean()),
                    "min": float(confidence_scores.min()),
                    "max": float(confidence_scores.max())
                }

                # Generate inference info
                info_message = (
                    f"Inference completed on {metrics['total_samples']} samples\n"
                    f"Processing time: {metrics['processing_time']:.3f}s\n"
                    f"Average confidence: {metrics['confidence_stats']['mean']:.3f}\n"
                    f"Output shape: {list(output.shape)}"
                )

                # Return results
                return (
                    output.cpu(),
                    confidence_scores.cpu() if return_confidence == "True" else torch.empty(0),
                    info_message,
                    metrics
                )

        except Exception as e:
            error_message = f"Error during inference: {str(e)}"
            return (torch.empty(0), torch.empty(0), error_message, {})


class NntVisualizeConfidenceScores:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "confidence_scores": ("TENSOR",),
                "image_width": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 64
                }),
                "image_height": ("INT", {
                    "default": 768,
                    "min": 256,
                    "max": 4096,
                    "step": 64
                }),
                "plot_type": (["histogram", "scatter", "box", "combined"], {
                    "default": "combined"
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "visualize_confidence"
    OUTPUT_NODE = True
    CATEGORY = "NNT Neural Network Toolkit/Inference"

    def visualize_confidence(self, confidence_scores, image_width, image_height, plot_type, threshold):
        try:
            import torch
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            from PIL import Image
            import io
            
            # Setup figure with the right size
            dpi = 100
            fig_width = image_width / dpi
            fig_height = image_height / dpi
            
            if plot_type == "combined":
                fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(fig_width, fig_height), dpi=dpi)
                
                # Histogram
                sns.histplot(confidence_scores.numpy(), bins=30, ax=ax1)
                ax1.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
                ax1.set_title('Confidence Score Distribution')
                ax1.set_xlabel('Confidence Score')
                ax1.set_ylabel('Count')
                ax1.legend()
                
                # Scatter plot
                ax2.scatter(range(len(confidence_scores)), confidence_scores.numpy(), alpha=0.5)
                ax2.axhline(y=threshold, color='r', linestyle='--')
                ax2.set_title('Confidence Scores per Sample')
                ax2.set_xlabel('Sample Index')
                ax2.set_ylabel('Confidence Score')
                
                # Box plot
                sns.boxplot(y=confidence_scores.numpy(), ax=ax3)
                ax3.axhline(y=threshold, color='r', linestyle='--')
                ax3.set_title('Confidence Score Statistics')
                ax3.set_ylabel('Confidence Score')
                
            else:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
                
                if plot_type == "histogram":
                    sns.histplot(confidence_scores.numpy(), bins=30, ax=ax)
                    ax.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
                    ax.set_title('Confidence Score Distribution')
                    ax.set_xlabel('Confidence Score')
                    ax.set_ylabel('Count')
                    ax.legend()
                
                elif plot_type == "scatter":
                    ax.scatter(range(len(confidence_scores)), confidence_scores.numpy(), alpha=0.5)
                    ax.axhline(y=threshold, color='r', linestyle='--')
                    ax.set_title('Confidence Scores per Sample')
                    ax.set_xlabel('Sample Index')
                    ax.set_ylabel('Confidence Score')
                
                elif plot_type == "box":
                    sns.boxplot(y=confidence_scores.numpy(), ax=ax)
                    ax.axhline(y=threshold, color='r', linestyle='--')
                    ax.set_title('Confidence Score Statistics')
                    ax.set_ylabel('Confidence Score')
            
            plt.tight_layout()
            
            # Convert plot to tensor
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            
            image = Image.open(buf).convert('RGB')
            image_tensor = torch.tensor(np.array(image).astype(np.float32) / 255.0)
            image_tensor = torch.unsqueeze(image_tensor, 0)
            
            plt.close(fig)
            buf.close()
            
            # Generate summary statistics
            stats_summary = f"""Confidence Score Statistics:
Mean: {confidence_scores.mean():.3f}
Median: {confidence_scores.median():.3f}
Std Dev: {confidence_scores.std():.3f}
Min: {confidence_scores.min():.3f}
Max: {confidence_scores.max():.3f}
Samples above threshold: {(confidence_scores > threshold).sum()}/{len(confidence_scores)}
"""
            
            return (image_tensor, stats_summary)
            
        except Exception as e:
            error_msg = f"Error visualizing confidence scores: {str(e)}"
            empty_tensor = torch.zeros((1, image_height, image_width, 3), dtype=torch.float32)
            return (empty_tensor, error_msg)

class NntAnalyzeInferenceMetrics:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "metrics": ("DICT",),
                "image_width": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 4096,
                    "step": 64
                }),
                "image_height": ("INT", {
                    "default": 768,
                    "min": 256,
                    "max": 4096,
                    "step": 64
                }),
                "plot_type": (["performance", "confidence", "combined"], {
                    "default": "combined"
                })
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    FUNCTION = "analyze_metrics"
    OUTPUT_NODE = True
    CATEGORY = "NNT Neural Network Toolkit/Inference"

    def analyze_metrics(self, metrics, image_width, image_height, plot_type):
        try:
            import torch
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            from PIL import Image
            import io
            
            # Setup figure with the right size
            dpi = 100
            fig_width = image_width / dpi
            fig_height = image_height / dpi
            
            if plot_type == "combined":
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height), dpi=dpi)
                
                # Performance metrics
                if "batch_metrics" in metrics:
                    times = [m.get("processing_time", 0) for m in metrics["batch_metrics"]]
                    ax1.plot(times, marker='o')
                    ax1.set_title('Processing Time per Batch')
                    ax1.set_xlabel('Batch')
                    ax1.set_ylabel('Time (seconds)')
                    ax1.grid(True)
                
                # Confidence statistics
                if "confidence_stats" in metrics:
                    stats = metrics["confidence_stats"]
                    stats_data = [stats.get("min", 0), stats.get("mean", 0), stats.get("max", 0)]
                    ax2.bar(['Min', 'Mean', 'Max'], stats_data)
                    ax2.set_title('Confidence Score Statistics')
                    ax2.set_ylabel('Confidence Value')
                
            else:
                fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
                
                if plot_type == "performance" and "batch_metrics" in metrics:
                    times = [m.get("processing_time", 0) for m in metrics["batch_metrics"]]
                    ax.plot(times, marker='o')
                    ax.set_title('Processing Time per Batch')
                    ax.set_xlabel('Batch')
                    ax.set_ylabel('Time (seconds)')
                    ax.grid(True)
                
                elif plot_type == "confidence" and "confidence_stats" in metrics:
                    stats = metrics["confidence_stats"]
                    stats_data = [stats.get("min", 0), stats.get("mean", 0), stats.get("max", 0)]
                    ax.bar(['Min', 'Mean', 'Max'], stats_data)
                    ax.set_title('Confidence Score Statistics')
                    ax.set_ylabel('Confidence Value')
            
            plt.tight_layout()
            
            # Convert plot to tensor
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            
            image = Image.open(buf).convert('RGB')
            image_tensor = torch.tensor(np.array(image).astype(np.float32) / 255.0)
            image_tensor = torch.unsqueeze(image_tensor, 0)
            
            plt.close(fig)
            buf.close()
            
            # Generate detailed metrics report
            report = f"""Inference Metrics Summary:
Total Samples: {metrics.get('total_samples', 0)}
Total Processing Time: {metrics.get('processing_time', 0):.3f} seconds
Samples per Second: {metrics.get('total_samples', 0) / max(metrics.get('processing_time', 1), 1e-6):.1f}

Confidence Statistics:
Min: {metrics.get('confidence_stats', {}).get('min', 0):.3f}
Mean: {metrics.get('confidence_stats', {}).get('mean', 0):.3f}
Max: {metrics.get('confidence_stats', {}).get('max', 0):.3f}
"""
            
            return (image_tensor, report)
            
        except Exception as e:
            error_msg = f"Error analyzing metrics: {str(e)}"
            empty_tensor = torch.zeros((1, image_height, image_width, 3), dtype=torch.float32)
            return (empty_tensor, error_msg)

