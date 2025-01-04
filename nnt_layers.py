# dynamic_model_builder.py

import torch
import torch.nn as nn
import os


PADDING_MODES = ["zeros", "reflect", "replicate", "circular"]
NONLINEARITY_TYPES = ["relu", "leaky_relu", "selu", "tanh", "linear", "sigmoid"]
INIT_MODES = ["fan_in", "fan_out"]
LAYER_NORM_TYPES = ["None", "BatchNorm", "LayerNorm"]

NORMALIZATION_TYPES = [
    "None",
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "LayerNorm",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "GroupNorm",
    "LocalResponseNorm"
]

# List of activation function names
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

BIAS_INIT_METHODS = [
    "default",
    "zeros",
    "ones",
    "normal",
    "uniform",
]

PADDING_MODES = [
    "zeros",
    "reflect",
    "replicate",
    "circular"
]

NONLINEARITY_TYPES = [
    "relu",
    "leaky_relu",
    "selu", 
    "tanh",
    "linear",
    "sigmoid"
]

INIT_MODES = [
    "fan_in",
    "fan_out"
]

LAYER_NORM_TYPES = [
    "None",
    "BatchNorm",
    "LayerNorm",
    "InstanceNorm",
    "GroupNorm",
    "LocalResponseNorm"
]

POOLING_TYPES = [
    "MaxPool1d", "MaxPool2d", "MaxPool3d",
    "AvgPool1d", "AvgPool2d", "AvgPool3d",
    "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d",
    "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d",
    "MaxUnpool1d", "MaxUnpool2d", "MaxUnpool3d",
    "FractionalMaxPool2d", "LPPool1d", "LPPool2d"
]

CONV_TYPES = [
    "Conv1d", "Conv2d", "Conv3d",
    "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d",
    "Unfold", "Fold"
]

RNN_NONLINEARITY_TYPES = [
    "tanh",
    "relu"
]

# Remove duplicate lists and update references
NONLINEARITY_TYPES_LIST = NONLINEARITY_TYPES
WEIGHT_INIT_FUNCTIONS_LIST = WEIGHT_INIT_METHODS
BIAS_INIT_FUNCTIONS_LIST = BIAS_INIT_METHODS

class NntDefineDenseLayer:
    """
    Advanced version of the dense layer node with comprehensive configuration options.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_nodes": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 8192,
                    "step": 1,
                    "display": "number"
                }),
                "activation_function": (ACTIVATION_FUNCTIONS, {"default": "ReLU"}),
                "use_bias": (["True", "False"], {"default": "True"}),
                # Weight initialization parameters
                "weight_init": (WEIGHT_INIT_FUNCTIONS_LIST, {"default": "kaiming_normal"}),
                "weight_init_gain": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 10.0,
                    "step": 0.01
                }),
                "weight_init_mode": (INIT_MODES, {"default": "fan_in"}),
                "weight_init_nonlinearity": (NONLINEARITY_TYPES, {"default": "relu"}),
                # Bias initialization
                "bias_init": (BIAS_INIT_FUNCTIONS_LIST, {"default": "zeros"}),
                "bias_init_value": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                # Normalization
                "normalization": (LAYER_NORM_TYPES, {"default": "None"}),
                "norm_eps": ("FLOAT", {
                    "default": 1e-5,
                    "min": 1e-12,
                    "max": 1e-3,
                    "step": 1e-6
                }),
                "norm_momentum": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.001,
                    "max": 0.999,
                    "step": 0.001
                }),
                "norm_affine": (["True", "False"], {"default": "True"}),
                # Regularization
                "dropout_rate": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.9,
                    "step": 0.1
                }),
                "alpha": ("FLOAT", {  # For LeakyReLU, ELU, etc.
                    "default": 0.01,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "num_copies": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),
            },
            "optional": {
                "LAYER_STACK": ("LIST",),
            },
        }

    RETURN_TYPES = ("LIST", "INT")
    RETURN_NAMES = ("LAYER_STACK", "num_nodes")
    FUNCTION = "define_dense_layer"
    CATEGORY = "NNT Neural Network Toolkit/Layers"

    def define_dense_layer(self, num_nodes, activation_function, use_bias, 
                          weight_init, weight_init_gain, weight_init_mode, weight_init_nonlinearity,
                          bias_init, bias_init_value, normalization, norm_eps, norm_momentum, 
                          norm_affine, dropout_rate, alpha, num_copies, LAYER_STACK=None):
        # Initialize or copy the layer stack
        if LAYER_STACK is None:
            LAYER_STACK = []
        else:
            LAYER_STACK = LAYER_STACK.copy()

        # Create the layer definition with all advanced parameters
        layer = {
            'type': 'Linear',
            'num_nodes': num_nodes,
            'activation': activation_function,
            'use_bias': use_bias == 'True',
            
            # Weight initialization
            'weight_init': weight_init,
            'weight_init_gain': float(weight_init_gain),
            'weight_init_mode': weight_init_mode,
            'weight_init_nonlinearity': weight_init_nonlinearity,
            
            # Bias initialization
            'bias_init': bias_init,
            'bias_init_value': float(bias_init_value),
            
            # Normalization
            'normalization': normalization,
            'norm_eps': float(norm_eps),
            'norm_momentum': float(norm_momentum),
            'norm_affine': norm_affine == 'True',
            
            # Regularization and activation parameters
            'dropout_rate': float(dropout_rate),
            'alpha': float(alpha),  # For activation functions that need it
        }

        # Append the layer definition 'num_copies' times
        for _ in range(num_copies):
            LAYER_STACK.append(layer.copy())

        return (LAYER_STACK, num_nodes)

class NntDefineGRULayer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_size": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 2048,
                    "step": 1
                }),
                "hidden_size": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 2048,
                    "step": 1
                }),
                "num_layers": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
                "bias": (["True", "False"], {"default": "True"}),
                "batch_first": (["True", "False"], {"default": "True"}),
                "dropout": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "bidirectional": (["True", "False"], {"default": "False"}),
            },
            "optional": {
                "LAYER_STACK": ("LIST",),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("LAYER_STACK",)
    FUNCTION = "define_gru_layer"
    CATEGORY = "NNT Neural Network Toolkit/Layers"

    def define_gru_layer(self, input_size, hidden_size, num_layers, bias, 
                        batch_first, dropout, bidirectional, LAYER_STACK=None):
        if LAYER_STACK is None:
            LAYER_STACK = []
        else:
            LAYER_STACK = LAYER_STACK.copy()

        layer = {
            'type': 'GRU',
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'bias': bias == 'True',
            'batch_first': batch_first == 'True',
            'dropout': dropout,
            'bidirectional': bidirectional == 'True'
        }

        LAYER_STACK.append(layer)
        return (LAYER_STACK,)

class NntDefineRNNLayer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_size": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 2048,
                    "step": 1
                }),
                "hidden_size": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 2048,
                    "step": 1
                }),
                "num_layers": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
                "nonlinearity": (RNN_NONLINEARITY_TYPES, {"default": "tanh"}),
                "bias": (["True", "False"], {"default": "True"}),
                "batch_first": (["True", "False"], {"default": "True"}),
                "dropout": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "bidirectional": (["True", "False"], {"default": "False"}),
            },
            "optional": {
                "LAYER_STACK": ("LIST",),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("LAYER_STACK",)
    FUNCTION = "define_rnn_layer"
    CATEGORY = "NNT Neural Network Toolkit/Layers"

    def define_rnn_layer(self, input_size, hidden_size, num_layers, nonlinearity,
                        bias, batch_first, dropout, bidirectional, LAYER_STACK=None):
        if LAYER_STACK is None:
            LAYER_STACK = []
        else:
            LAYER_STACK = LAYER_STACK.copy()

        layer = {
            'type': 'RNN',
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'nonlinearity': nonlinearity,
            'bias': bias == 'True',
            'batch_first': batch_first == 'True',
            'dropout': dropout,
            'bidirectional': bidirectional == 'True'
        }

        LAYER_STACK.append(layer)
        return (LAYER_STACK,)


class NntDefineConvLayer:
    """Node for defining convolutional layers with full configuration options and hyperparameter support."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conv_type": (CONV_TYPES, {
                    "default": "Conv2d"
                }),
                "out_channels": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 2048,
                    "step": 1
                }),
                "kernel_size": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 15,
                    "step": 1
                }),
                "stride": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1
                }),
                "padding": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 10,
                    "step": 1
                }),
                "padding_mode": (PADDING_MODES, {"default": "zeros"}),
                "output_padding": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2,
                    "step": 1
                }),
                "dilation": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5,
                    "step": 1
                }),
                "groups": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 2048,
                    "step": 1
                }),
                "use_bias": (["True", "False"], {"default": "True"}),
                "activation_function": (ACTIVATION_FUNCTIONS, {"default": "ReLU"}),
                "normalization": (NORMALIZATION_TYPES, {"default": "BatchNorm"}),
                "norm_eps": ("FLOAT", {
                    "default": 1e-5,
                    "min": 1e-12,
                    "max": 1e-3,
                    "step": 1e-6
                }),
                "norm_momentum": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.001,
                    "max": 0.999,
                    "step": 0.001
                }),
                "norm_affine": (["True", "False"], {"default": "True"}),
                "dropout_rate": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 0.9,
                    "step": 0.1
                }),
                "weight_init": (WEIGHT_INIT_METHODS, {"default": "kaiming_normal"}),
                "weight_init_gain": ("FLOAT", {
                    "default": 1.414,
                    "min": 0.01,
                    "max": 10.0,
                    "step": 0.01
                }),
                "weight_init_mode": (INIT_MODES, {"default": "fan_out"}),
                "weight_init_nonlinearity": (NONLINEARITY_TYPES, {"default": "relu"}),
                "num_copies": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1
                })
            },
            "optional": {
                "LAYER_STACK": ("LIST",),
                "hyperparameters": ("DICT",)
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("layer_stack",)
    FUNCTION = "define_conv_layer"
    CATEGORY = "NNT Neural Network Toolkit/Layers"

    def define_conv_layer(self, conv_type, out_channels, kernel_size, stride, padding, 
                        padding_mode, output_padding, dilation, groups, use_bias, activation_function,
                        normalization, norm_eps, norm_momentum, norm_affine, dropout_rate,
                        weight_init, weight_init_gain, weight_init_mode, weight_init_nonlinearity,
                        num_copies, LAYER_STACK=None, hyperparameters=None):
        try:
            if LAYER_STACK is None:
                LAYER_STACK = []
            else:
                LAYER_STACK = LAYER_STACK.copy()

            # Create the layers
            for _ in range(num_copies):
                # Main convolution layer definition
                conv_layer = {
                    'type': conv_type,
                    'in_channels': 3 if not LAYER_STACK else LAYER_STACK[-1].get('out_channels', 3),
                    'out_channels': out_channels,
                    'kernel_size': kernel_size,
                    'stride': stride,
                    'padding': padding,
                    'padding_mode': padding_mode,
                    'dilation': dilation,
                    'groups': groups,
                    'bias': use_bias == "True",
                    'weight_init': weight_init,
                    'weight_init_gain': float(weight_init_gain),
                    'weight_init_mode': weight_init_mode,
                    'weight_init_nonlinearity': weight_init_nonlinearity
                }

                # Add output_padding only for ConvTranspose layers
                if 'ConvTranspose' in conv_type:
                    conv_layer['output_padding'] = output_padding

                LAYER_STACK.append(conv_layer)

                # Normalization layer
                if normalization != "None":
                    norm_layer = {
                        'type': normalization,
                        'num_features': out_channels,
                        'out_channels': out_channels,
                        'eps': float(norm_eps),
                        'momentum': float(norm_momentum),
                        'affine': norm_affine == "True",
                        'track_running_stats': True
                    }
                    LAYER_STACK.append(norm_layer)

                # Activation layer
                if activation_function != "None":
                    act_layer = {
                        'type': 'Activation',
                        'activation_type': activation_function,
                        'out_channels': out_channels
                    }
                    LAYER_STACK.append(act_layer)

                # Dropout layer
                if dropout_rate > 0:
                    dropout_layer = {
                        'type': 'Dropout',
                        'dropout_rate': float(dropout_rate),
                        'out_channels': out_channels
                    }
                    LAYER_STACK.append(dropout_layer)

                return (LAYER_STACK,)

        except Exception as e:
            import traceback
            error_msg = f"Error defining convolutional layer: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return (LAYER_STACK if LAYER_STACK is not None else [],)
        
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # This ensures the node is evaluated when hyperparameters change
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Basic input validation
        try:
            if kwargs.get('stride', 1) > kwargs.get('kernel_size', 3):
                return "Stride should not be larger than kernel size"
            return True
        except Exception as e:
            return f"Validation error: {str(e)}"


class NntDefinePoolingLayer:
    @classmethod
    def INPUT_TYPES(cls):
        POOL_TYPES = [
            "MaxPool1d", "MaxPool2d", "MaxPool3d",
            "AvgPool1d", "AvgPool2d", "AvgPool3d",
            "AdaptiveMaxPool1d", "AdaptiveMaxPool2d", "AdaptiveMaxPool3d",
            "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d",
            "MaxUnpool1d", "MaxUnpool2d", "MaxUnpool3d",
            "FractionalMaxPool2d", "LPPool1d", "LPPool2d"
        ]
        
        return {
            "required": {
                "pooling_type": (POOL_TYPES, {"default": "MaxPool2d"}),
                # Basic pooling parameters - handled as string for multi-dimensional input
                "kernel_size": ("STRING", {
                    "default": "2",
                    "multiline": False,
                    "placeholder": "Single int or tuple: 2 or (2,2) or (2,2,2)"
                }),
                "stride": ("STRING", {
                    "default": "2",
                    "multiline": False,
                    "placeholder": "Single int or tuple"
                }),
                "padding": ("STRING", {
                    "default": "0",
                    "multiline": False,
                    "placeholder": "Single int or tuple"
                }),
                # Advanced MaxPool options
                "dilation": ("STRING", {
                    "default": "1",
                    "multiline": False,
                    "placeholder": "Single int or tuple"
                }),
                "ceil_mode": (["True", "False"], {"default": "False"}),
                "return_indices": (["True", "False"], {"default": "False"}),
                # AvgPool specific options
                "count_include_pad": (["True", "False"], {"default": "True"}),
                # Adaptive pooling options - handled as string for multi-dimensional output size
                "output_size": ("STRING", {
                    "default": "1",
                    "multiline": False,
                    "placeholder": "Single int or tuple"
                }),
                # FractionalMaxPool options
                "fractional_factor": ("FLOAT", {
                    "default": 1.5,
                    "min": 1.0,
                    "max": 3.0,
                    "step": 0.1
                }),
                # LPPool options
                "norm_type": ("FLOAT", {
                    "default": 2.0,
                    "min": 1.0,
                    "max": 6.0,
                    "step": 0.1
                }),
                "flatten_output": (["True", "False"], {"default": "False"}),
                "num_copies": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
            },
            "optional": {
                "LAYER_STACK": ("LIST",),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("LAYER_STACK",)
    FUNCTION = "define_pooling_layer"
    CATEGORY = "NNT Neural Network Toolkit/Layers"

    def parse_size_param(self, param_str, dim=2):
        try:
            val = eval(param_str)
            if isinstance(val, (int, float)):
                return val
            if isinstance(val, (tuple, list)):
                if len(val) == dim:
                    return val
            raise ValueError
        except:
            if 'd1' in self.pooling_type:
                return int(param_str)
            elif 'd2' in self.pooling_type:
                return (int(param_str), int(param_str))
            else:  # 3d
                return (int(param_str), int(param_str), int(param_str))

    def define_pooling_layer(self, pooling_type, kernel_size, stride, padding, dilation, 
                            ceil_mode, return_indices, count_include_pad, output_size,
                            fractional_factor, norm_type, flatten_output, num_copies, LAYER_STACK=None):
        if LAYER_STACK is None:
            LAYER_STACK = []
        else:
            LAYER_STACK = LAYER_STACK.copy()

        # Determine dimensionality
        dim = 2  # default
        if '1d' in pooling_type:
            dim = 1
        elif '3d' in pooling_type:
            dim = 3

        # Parse size parameters
        kernel = self.parse_size_param(kernel_size, dim)
        stride_val = self.parse_size_param(stride, dim)
        padding_val = self.parse_size_param(padding, dim)
        dilation_val = self.parse_size_param(dilation, dim)
        output_size_val = self.parse_size_param(output_size, dim)

        layer = {
            'type': pooling_type,
            'flatten_output': flatten_output == 'True'
        }

        if pooling_type.startswith(('MaxPool', 'AvgPool', 'LPPool')):
            layer.update({
                'kernel_size': kernel,
                'stride': stride_val,
                'padding': padding_val,
                'ceil_mode': ceil_mode == 'True'
            })

            if 'MaxPool' in pooling_type:
                layer.update({
                    'dilation': dilation_val,
                    'return_indices': return_indices == 'True'
                })
            elif 'AvgPool' in pooling_type:
                layer['count_include_pad'] = count_include_pad == 'True'
            elif 'LPPool' in pooling_type:
                layer['norm_type'] = float(norm_type)

        elif pooling_type.startswith('Adaptive'):
            layer.update({
                'output_size': output_size_val,
                'return_indices': return_indices == 'True' if 'Max' in pooling_type else False
            })

        elif pooling_type.startswith('MaxUnpool'):
            layer.update({
                'kernel_size': kernel,
                'stride': stride_val,
                'padding': padding_val
            })

        elif pooling_type == 'FractionalMaxPool2d':
            layer.update({
                'kernel_size': kernel,
                'fractional_factor': float(fractional_factor),
                'return_indices': return_indices == 'True'
            })

        for _ in range(num_copies):
            LAYER_STACK.append(layer.copy())

        return (LAYER_STACK,)
    
class NntInputLayer:
    """
    A node to define the input layer with the specified input shape provided as a text field.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_shape_str": ("STRING", {
                    "default": "[3, 224, 224]",
                    "multiline": False,
                    "placeholder": "Enter input shape as a list, e.g., [3, 224, 224]"
                }),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("LAYER_STACK",)
    FUNCTION = "define_input_layer"
    CATEGORY = "NNT Neural Network Toolkit/Layers"

    def define_input_layer(self, input_shape_str):
        # Parse the input_shape_str into a list of integers
        try:
            # Remove any whitespace and parse the string
            input_shape = eval(input_shape_str.strip(), {"__builtins__": None}, {})
            if not isinstance(input_shape, list):
                raise ValueError("Input shape must be a list of integers.")
            # Validate that all elements are integers
            for dim in input_shape:
                if not isinstance(dim, int):
                    raise ValueError("All dimensions must be integers.")
            # Ensure input_shape is not empty
            if len(input_shape) == 0:
                raise ValueError("Input shape cannot be empty.")
        except Exception as e:
            raise ValueError(f"Invalid input_shape format: {str(e)}")

        # Create the layer definition
        layer = {
            'type': 'Input',
            'input_shape': input_shape,
        }
        LAYER_STACK = [layer]
        return (LAYER_STACK,)
    
class NntDefineNormLayer:
    """
    Node for defining normalization layers.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "norm_type": (LAYER_NORM_TYPES, {
                    "default": "BatchNorm2d"
                }),
                "num_features": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 2048,
                    "step": 1
                }),
                "eps": ("FLOAT", {
                    "default": 1e-5,
                    "min": 1e-10,
                    "max": 1e-3,
                    "step": 1e-6
                }),
                "momentum": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "affine": (["True", "False"], {
                    "default": "True"
                }),
                "track_running_stats": (["True", "False"], {
                    "default": "True"
                }),
            },
            "optional": {
                "LAYER_STACK": ("LIST",),
            },
        }

    RETURN_TYPES = ("LIST",) 
    RETURN_NAMES = ("LAYER_STACK",)
    FUNCTION = "define_norm_layer"
    CATEGORY = "NNT Neural Network Toolkit/Layers"

    def define_norm_layer(self, norm_type, num_features, eps, momentum, 
                         affine, track_running_stats, LAYER_STACK=None):
        # Initialize or copy the layer stack
        if LAYER_STACK is None:
            LAYER_STACK = []
        else:
            LAYER_STACK = LAYER_STACK.copy()

        # Create the layer definition
        layer = {
            'type': norm_type,
            'num_features': num_features,
            'eps': float(eps),
            'momentum': float(momentum),
            'affine': affine == 'True',
            'track_running_stats': track_running_stats == 'True'
        }

        # Append the layer definition
        LAYER_STACK.append(layer.copy())

        return (LAYER_STACK,)
    
class NntDefineActivationLayer:
    """
    Node for defining activation layers.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "activation_type": (ACTIVATION_FUNCTIONS, {
                    "default": "ReLU"
                }),
                "inplace": (["True", "False"], {
                    "default": "False"
                }),
                "negative_slope": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "num_parameters": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 2048,
                    "step": 1
                }),
                "alpha": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "LAYER_STACK": ("LIST",),
            },
        }

    RETURN_TYPES = ("LIST",)  
    RETURN_NAMES = ("LAYER_STACK",)
    FUNCTION = "define_activation_layer"
    CATEGORY = "NNT Neural Network Toolkit/Layers"

    def define_activation_layer(self, activation_type, inplace, negative_slope,
                              num_parameters, alpha, LAYER_STACK=None):
        # Initialize or copy the layer stack
        if LAYER_STACK is None:
            LAYER_STACK = []
        else:
            LAYER_STACK = LAYER_STACK.copy()

        # Create the layer definition
        layer = {
            'type': 'Activation',
            'activation_type': activation_type,
            'inplace': inplace == 'True',
            'negative_slope': float(negative_slope),
            'num_parameters': num_parameters,
            'alpha': float(alpha)
        }

        # Append the layer definition
        LAYER_STACK.append(layer.copy())

        return (LAYER_STACK,)

class NntDefineLSTMLayer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_size": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 2048,
                    "step": 1
                }),
                "hidden_size": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 2048,
                    "step": 1
                }),
                "num_layers": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
                "bias": (["True", "False"], {"default": "True"}),
                "batch_first": (["True", "False"], {"default": "True"}),
                "dropout": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "bidirectional": (["True", "False"], {"default": "False"}),
            },
            "optional": {
                "LAYER_STACK": ("LIST",),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("LAYER_STACK",)
    FUNCTION = "define_lstm_layer"
    CATEGORY = "NNT Neural Network Toolkit/Layers"

    def define_lstm_layer(self, input_size, hidden_size, num_layers, bias, 
                         batch_first, dropout, bidirectional, LAYER_STACK=None):
        if LAYER_STACK is None:
            LAYER_STACK = []
        else:
            LAYER_STACK = LAYER_STACK.copy()

        layer = {
            'type': 'LSTM',
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'bias': bias == 'True',
            'batch_first': batch_first == 'True',
            'dropout': dropout,
            'bidirectional': bidirectional == 'True'
        }

        LAYER_STACK.append(layer)
        return (LAYER_STACK,)

class NntShowLayerStack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LAYER_STACK": ("LIST",),
                "show_details": (["Basic", "Detailed"], {"default": "Basic"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("layer_stack_report")
    FUNCTION = "show_layer_stack"
    OUTPUT_NODE = True
    CATEGORY = "NNT Neural Network Toolkit/Layers"

    def format_layer_params(self, params):
        """Helper function to format parameter dictionaries."""
        return ", ".join(f"{k}={v}" for k, v in params.items() if v is not None and v != "None" and v != "" and str(v) != "0.0")

    def show_layer_stack(self, LAYER_STACK, show_details):
        stack_info = []
        
        for idx, layer in enumerate(LAYER_STACK):
            layer_type = layer.get('type', 'Unknown')
            basic_info = [f"\nLayer {idx + 1}: Type={layer_type}"]
            
            # Input Layer
            if layer_type == 'Input':
                basic_info.append(f"Shape={layer.get('input_shape', 'N/A')}")

            # Dense/Linear Layer
            elif layer_type == 'Linear':
                basic_info.extend([
                    f"Nodes={layer.get('num_nodes', 'N/A')}",
                    f"Activation={layer.get('activation', 'N/A')}"
                ])
                if show_details == "Detailed":
                    detailed_info = {
                        'Use Bias': layer.get('use_bias'),
                        'Weight Init': layer.get('weight_init'),
                        'Init Gain': layer.get('weight_init_gain'),
                        'Init Mode': layer.get('weight_init_mode'),
                        'Init NonLin': layer.get('weight_init_nonlinearity'),
                        'Normalization': layer.get('normalization'),
                        'Norm Eps': layer.get('norm_eps'),
                        'Norm Momentum': layer.get('norm_momentum'),
                        'Norm Affine': layer.get('norm_affine'),
                        'Dropout': layer.get('dropout_rate')
                    }
                    basic_info.append(f"Details: {self.format_layer_params(detailed_info)}")

            # Convolutional Layers
            elif layer_type in ['Conv1d', 'Conv2d', 'Conv3d', 'ConvTranspose1d', 'ConvTranspose2d', 'ConvTranspose3d']:
                basic_info.extend([
                    f"Out Channels={layer.get('out_channels', 'N/A')}",
                    f"Kernel={layer.get('kernel_size', 'N/A')}",
                    f"Stride={layer.get('stride', 'N/A')}",
                    f"Padding={layer.get('padding', 'N/A')}"
                ])
                if show_details == "Detailed":
                    detailed_info = {
                        'Padding Mode': layer.get('padding_mode'),
                        'Dilation': layer.get('dilation'),
                        'Groups': layer.get('groups'),
                        'Output Padding': layer.get('output_padding'),
                        'Activation': layer.get('activation'),
                        'Use Bias': layer.get('use_bias'),
                        'Weight Init': layer.get('weight_init'),
                        'Init Gain': layer.get('weight_init_gain'),
                        'Normalization': layer.get('normalization'),
                        'Dropout': layer.get('dropout_rate')
                    }
                    basic_info.append(f"Details: {self.format_layer_params(detailed_info)}")

            # Pooling Layers
            elif any(x in layer_type for x in ['Pool1d', 'Pool2d', 'Pool3d']):
                basic_info.extend([
                    f"Type={layer_type}",
                    f"Kernel={layer.get('kernel_size', 'N/A')}",
                    f"Stride={layer.get('stride', 'N/A')}"
                ])
                if show_details == "Detailed":
                    detailed_info = {
                        'Padding': layer.get('padding'),
                        'Dilation': layer.get('dilation'),
                        'Ceil Mode': layer.get('ceil_mode'),
                        'Return Indices': layer.get('return_indices'),
                        'Count Include Pad': layer.get('count_include_pad'),
                        'Output Size': layer.get('output_size'),
                        'Flatten Output': layer.get('flatten_output')
                    }
                    basic_info.append(f"Details: {self.format_layer_params(detailed_info)}")

            # Recurrent Layers (LSTM, GRU, RNN)
            elif layer_type in ['LSTM', 'GRU', 'RNN']:
                basic_info.extend([
                    f"Input Size={layer.get('input_size', 'N/A')}",
                    f"Hidden Size={layer.get('hidden_size', 'N/A')}",
                    f"Num Layers={layer.get('num_layers', 'N/A')}"
                ])
                if show_details == "Detailed":
                    detailed_info = {
                        'Bias': layer.get('bias'),
                        'Batch First': layer.get('batch_first'),
                        'Dropout': layer.get('dropout'),
                        'Bidirectional': layer.get('bidirectional'),
                        'Nonlinearity': layer.get('nonlinearity')  # For RNN
                    }
                    basic_info.append(f"Details: {self.format_layer_params(detailed_info)}")

            # Normalization Layers
            elif any(x in layer_type for x in ['BatchNorm', 'LayerNorm', 'InstanceNorm', 'GroupNorm']):
                basic_info.extend([
                    f"Num Features={layer.get('num_features', 'N/A')}"
                ])
                if show_details == "Detailed":
                    detailed_info = {
                        'Eps': layer.get('eps'),
                        'Momentum': layer.get('momentum'),
                        'Affine': layer.get('affine'),
                        'Track Running Stats': layer.get('track_running_stats')
                    }
                    basic_info.append(f"Details: {self.format_layer_params(detailed_info)}")

            # Activation Layers
            elif layer_type == 'Activation':
                basic_info.extend([
                    f"Function={layer.get('activation_type', 'N/A')}"
                ])
                if show_details == "Detailed":
                    detailed_info = {
                        'Inplace': layer.get('inplace'),
                        'Negative Slope': layer.get('negative_slope'),
                        'Num Parameters': layer.get('num_parameters'),
                        'Alpha': layer.get('alpha')
                    }
                    basic_info.append(f"Details: {self.format_layer_params(detailed_info)}")


            stack_info.append(" | ".join(basic_info))

        if not stack_info:
            return ("No layers defined yet.",)

        return ("\n".join(stack_info),)

class NntDefineFlattenLayer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "LAYER_STACK": ("LIST",),
            }
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "define_flatten_layer"
    CATEGORY = "NNT Neural Network Toolkit/Layers"

    def define_flatten_layer(self, LAYER_STACK=None):
        if LAYER_STACK is None:
            LAYER_STACK = []
        else:
            LAYER_STACK = LAYER_STACK.copy()

        layer = {
            'type': 'Flatten',
            'start_dim': 1,
            'end_dim': -1
        }

        LAYER_STACK.append(layer)
        return (LAYER_STACK,)

class NntDefineReshapeLayer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_shape": ("STRING", {
                    "default": "[8,7,7]",
                    "multiline": False,
                    "placeholder": "e.g., [8,7,7] or [-1,8,7,7]"
                }),
            },
            "optional": {
                "LAYER_STACK": ("LIST",),
            },
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("LAYER_STACK",)
    FUNCTION = "define_reshape_layer"
    CATEGORY = "NNT Neural Network Toolkit/Layers"

    def define_reshape_layer(self, target_shape, LAYER_STACK=None):
        if LAYER_STACK is None:
            LAYER_STACK = []
        else:
            LAYER_STACK = LAYER_STACK.copy()

        # Parse the shape string into a list
        try:
            shape = eval(target_shape)
            if not isinstance(shape, (list, tuple)):
                raise ValueError("Target shape must be a list or tuple")
        except Exception as e:
            raise ValueError(f"Invalid shape format: {str(e)}")

        layer = {
            'type': 'Reshape',
            'target_shape': shape
        }

        LAYER_STACK.append(layer)
        return (LAYER_STACK,)