import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import ast

# Tensor data types
TENSOR_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
    "auto": "auto"  # Special case handling
}

# Tensor operations
TENSOR_OPERATIONS = [
    "add_tensors",
    "subtract_tensors",
    "multiply_tensors_elementwise",
    "matrix_multiply_tensors",
    "transpose_tensor",
    "inverse_tensor",
    "add_scalar_to_tensor",
    "multiply_tensor_by_scalar",
    "custom_function",
    "custom_function_with_grad",
    "gradient",
    "jacobian",
    "hessian",
    "gradient_norm"
]

# Data loading configurations
DATA_SOURCES = [
    "txt",
    "numpy",
    "python_pickle",
    "image_folder",
    "image_text_pairs",
    "text_text_pairs"
]


TOKENIZER_TYPES = [
    "basic",
    "wordpiece",
    "bpe"
]

IMAGE_INTERPOLATION_MODES = [
    "nearest",
    "bilinear",
    "bicubic"
]

NORMALIZE_RANGES = [
    "0-1",
    "-1-1",
    "standardize"
]

# Format options for tensor to text conversion
TEXT_FORMAT_OPTIONS = [
    "plain_text",
    "formatted_text",
    "summary"
]

# Configuration ranges
CONFIG_RANGES = {
    "precision": {
        "default": 4,
        "min": 0,
        "max": 10,
        "step": 1
    },
    "max_elements": {
        "default": 100,
        "min": 1,
        "max": 1000000,
        "step": 1
    },
    "image_size": {
        "default": 32,
        "min": 16,
        "max": 2048,
        "step": 1
    },
    "vocab_size": {
        "default": 10000,
        "min": 1,
        "max": 1000000,
        "step": 1
    },
    "sequence_length": {
        "default": 128,
        "min": 1,
        "max": 10000,
        "step": 1
    }
}

class NntTensorToText:
    """
    Node for converting a tensor into a text representation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("TENSOR",),
                "format_option": (TEXT_FORMAT_OPTIONS, {
                    "default": "plain_text"
                }),
                "precision": ("INT", CONFIG_RANGES["precision"]),
                "max_elements": ("INT", CONFIG_RANGES["max_elements"])
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_output",)
    FUNCTION = "tensor_to_text"
    OUTPUT_NODE = True
    CATEGORY = "NNT Neural Network Toolkit/Tensors"

    def tensor_to_text(self, tensor, format_option="plain_text", precision=4, max_elements=100):
        try:
            import torch
            import numpy as np

            # Convert tensor to numpy array
            np_array = tensor.detach().cpu().numpy()

            # Flatten the array if necessary
            total_elements = np_array.size
            if total_elements > max_elements:
                np_array = np_array.flatten()[:max_elements]
                truncated = True
            else:
                truncated = False

            np.set_printoptions(precision=precision, suppress=True)

            if format_option == "plain_text":
                text_output = np.array2string(np_array, separator=', ')
            elif format_option == "formatted_text":
                text_output = np.array2string(np_array, separator=', ', formatter={'float_kind':lambda x: f"{x:.{precision}f}"})
            elif format_option == "summary":
                mean_value = np_array.mean()
                std_value = np_array.std()
                min_value = np_array.min()
                max_value = np_array.max()
                shape = tensor.shape
                text_output = (
                    f"Tensor Summary:\n"
                    f"Shape: {shape}\n"
                    f"Mean: {mean_value:.{precision}f}\n"
                    f"Std: {std_value:.{precision}f}\n"
                    f"Min: {min_value:.{precision}f}\n"
                    f"Max: {max_value:.{precision}f}"
                )
            else:
                raise ValueError(f"Unsupported format option: {format_option}")

            if truncated and format_option != "summary":
                text_output += f"\n... (output truncated to first {max_elements} elements)"

            return (text_output,)
        except Exception as e:
            error_message = f"Error converting tensor to text: {str(e)}"
            return (error_message,)

import torch
import numpy as np

import torch
import numpy as np
import ast

class NntTextToTensor:
    """Convert text input to PyTorch tensor with configurable options"""
    
    CATEGORY = "NNT Neural Network Toolkit/Tensors"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dtype": (list(TENSOR_DTYPES.keys()), {
                    "default": "float32"
                }),
                "requires_grad": ("BOOLEAN", {"default": False}),
                "device": (["cpu", "cuda"],),
                "text_content": ("STRING", {"multiline": True}),  # Node UI input
            },
            "optional": {
                "input_text": ("STRING",{"forceInput":True}),  # Connector input
            }
        }
    
    RETURN_TYPES = ("TENSOR",)
    FUNCTION = "convert_text_to_tensor"
    OUTPUT_NODE = False
    
    DTYPE_MAP = TENSOR_DTYPES 
    
    def parse_list_string(self, text):
        try:
            # Remove any whitespace and newlines
            text = text.strip()
            # Safely evaluate the string as a Python literal
            data = ast.literal_eval(text)
            # Convert to numpy array
            return np.array(data, dtype=np.float32)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid list format. Expected format like [[1, 2, 3]] or [1, 2, 3]. Error: {str(e)}")
    
    def convert_text_to_tensor(self, text_content, dtype="auto", requires_grad=False, device="cpu", input_text=None):
        # Prioritize connector input over node input
        text = input_text if input_text is not None else text_content
        
        if text is None:
            raise ValueError("No text input provided")
            
        try:
            # Parse the input text as a Python list
            np_array = self.parse_list_string(text)
            
            # Determine dtype
            if dtype == "auto":
                # Check if all numbers are integers
                if np.all(np.equal(np.mod(np_array, 1), 0)):
                    torch_dtype = torch.int64
                else:
                    torch_dtype = torch.float32
            else:
                torch_dtype = self.DTYPE_MAP[dtype]
            
            # Convert to tensor
            tensor = torch.tensor(np_array, 
                                dtype=torch_dtype,
                                device=device,
                                requires_grad=requires_grad)
            
            return (tensor,)
            
        except ValueError as e:
            raise ValueError(f"Failed to convert text to tensor: {str(e)}")
        except Exception as e:
            raise Exception(f"Error processing input: {str(e)}")

import torch
import numpy as np
from PIL import Image

class NntTensorElementToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tensor": ("TENSOR",),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 99999,
                    "step": 1
                }),
                "convert_mode": (["L", "RGB"], {"default": "RGB"}),
                "clamp_range": (["True", "False"], {"default": "True"}),
                "reshape": (["True", "False"], {"default": "False"}),
                "channels": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 4,
                    "step": 1
                }),
                "height": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 8192,
                    "step": 1
                }),
                "width": ("INT", {
                    "default": 64,
                    "min": 1,
                    "max": 8192,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert_to_image"
    OUTPUT_NODE = True
    CATEGORY = "NNT Neural Network Toolkit/Tensors"

    def convert_to_image(self, tensor, index, convert_mode, clamp_range, reshape, channels, height, width):
        try:
            import torch
            import numpy as np
            from PIL import Image

            # Create empty tensor in case of errors - ensure 4D with float32
            empty_tensor = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
            
            if index >= len(tensor):
                return (empty_tensor,)

            # Get single item from batch
            image_tensor = tensor[index]

            # Handle reshaping of flattened tensors if requested
            if reshape and image_tensor.dim() == 1:
                total_elements = image_tensor.numel()
                expected_elements = channels * height * width
                if total_elements != expected_elements:
                    raise ValueError(f"Tensor has {total_elements} elements but expected {expected_elements} for reshaping")
                image_tensor = image_tensor.reshape(channels, height, width)
            
            # Ensure tensor has the right number of dimensions
            if image_tensor.dim() == 2:  # Single channel, H,W
                image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension
            elif image_tensor.dim() == 1:  # Flattened and not reshaped
                raise ValueError("Tensor is flattened. Enable reshape option to convert to image.")
            elif image_tensor.dim() != 3:
                raise ValueError(f"Expected 2D or 3D tensor, got {image_tensor.dim()}D")

            # Convert to float32 if needed
            image_tensor = image_tensor.float()

            # Clamp values if requested
            if clamp_range == "True":
                image_tensor = torch.clamp(image_tensor, 0, 1)

            # Convert to numpy and scale to 0-255 range
            image_array = (image_tensor.cpu().numpy() * 255).astype(np.uint8)

            # Handle channel order and convert to PIL
            if convert_mode == "L":
                if image_array.shape[0] == 3:  # RGB to grayscale
                    # Use standard RGB to grayscale conversion weights
                    image_array = np.dot(image_array.transpose(1, 2, 0), [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                else:
                    image_array = image_array[0]  # Take first channel
                pil_image = Image.fromarray(image_array, mode='L')
                # Convert back to tensor in ComfyUI format [B,C,H,W]
                result = torch.from_numpy(np.array(pil_image)[None,None,...]).float() / 255.0
            else:  # RGB
                if image_array.shape[0] != 3:
                    # If not 3 channels, create a 3-channel image by repeating the first channel
                    if image_array.shape[0] == 1:
                        image_array = np.repeat(image_array, 3, axis=0)
                    else:
                        raise ValueError(f"Expected 1 or 3 channels for RGB mode, got {image_array.shape[0]}")
                # Convert to HWC for PIL
                image_array = image_array.transpose(1, 2, 0)
                pil_image = Image.fromarray(image_array, mode='RGB')
                # Convert back to tensor in ComfyUI format [B,C,H,W]
                #result = torch.from_numpy(np.array(pil_image).transpose(2, 0, 1)[None,...]).float() / 255.0
                #result = torch.from_numpy(np.array(pil_image)).permute(1, 2, 0)
                result = torch.tensor(np.array(pil_image).astype(np.float32) / 255.0)
                result = torch.unsqueeze(result, 0)


            return (result,)

        except Exception as e:
            print(f"Error converting tensor to image: {str(e)}")
            return (empty_tensor,)


class NntRandomTensorGenerator:
    """
    Node for generating random tensors with various distributions and controls.
    Supports multiple random distributions with fine-grained control over parameters.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "distribution": ([
                    "uniform",
                    "normal",
                    "bernoulli",
                    "geometric",
                    "exponential",
                    "lognormal",
                    "cauchy",
                ], {
                    "default": "uniform"
                }),
                "data_shape": ("STRING", {
                    "default": "[100, 10]",
                    "multiline": False,
                    "placeholder": "e.g., [100, 10] for 100 samples with 10 features"
                }),
                "data_type": (["float32", "float64", "int32", "int64"], {
                    "default": "float32"
                }),
                # Distribution parameters
                "min_value": ("FLOAT", {
                    "default": 0.0,
                    "min": -1000.0,
                    "max": 1000.0,
                    "step": 0.1
                }),
                "max_value": ("FLOAT", {
                    "default": 1.0,
                    "min": -1000.0,
                    "max": 1000.0,
                    "step": 0.1
                }),
                "mean": ("FLOAT", {
                    "default": 0.0,
                    "min": -1000.0,
                    "max": 1000.0,
                    "step": 0.1
                }),
                "std": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1
                }),
                "rate": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1
                }),
                "requires_grad": (["True", "False"], {
                    "default": "True"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 99999999,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("TENSOR", "STRING", "INT")
    RETURN_NAMES = ("tensor", "info_message", "batch_size")
    FUNCTION = "generate_tensor"
    OUTPUT_NODE = True
    CATEGORY = "NNT Neural Network Toolkit/Tensors"

    def generate_tensor(self, distribution, data_shape, data_type, min_value, max_value,
                       mean, std, rate, requires_grad, seed):
        import torch
        
        try:
            # Set seed if provided
            if seed != -1:
                torch.manual_seed(seed)
                
            dtype_map = {
                'float32': torch.float32,
                'float64': torch.float64,
                'int32': torch.int32,
                'int64': torch.int64
            }
            torch_dtype = dtype_map[data_type]
            
            # Parse shape
            shape = eval(data_shape)
            if not isinstance(shape, (list, tuple)):
                raise ValueError("data_shape must be a list or tuple")
                
            # Generate random data based on distribution
            if distribution == "uniform":
                tensor = torch.empty(shape, dtype=torch_dtype).uniform_(min_value, max_value)
            elif distribution == "normal":
                tensor = torch.empty(shape, dtype=torch_dtype).normal_(mean, std)
            elif distribution == "bernoulli":
                p = (max_value - min_value) / 2 + min_value
                tensor = torch.empty(shape, dtype=torch_dtype).bernoulli_(p)
            elif distribution == "geometric":
                p = torch.clamp(torch.tensor(rate), 0, 1)
                tensor = torch.empty(shape, dtype=torch_dtype).geometric_(p)
            elif distribution == "exponential":
                tensor = torch.empty(shape, dtype=torch_dtype).exponential_(rate)
            elif distribution == "lognormal":
                tensor = torch.empty(shape, dtype=torch_dtype).log_normal_(mean, std)
            elif distribution == "cauchy":
                tensor = torch.empty(shape, dtype=torch_dtype).cauchy_(mean, std)
            else:
                raise ValueError(f"Unsupported distribution: {distribution}")

            # Convert to float for training if needed
            if tensor.dtype not in [torch.float32, torch.float64]:
                tensor = tensor.float()

            # Set requires_grad if requested
            if requires_grad == "True":
                tensor = tensor.requires_grad_(True)

            batch_size = shape[0] if len(shape) > 0 else 1
            info_message = (f"Generated {distribution} random tensor with shape {shape}\n"
                          f"dtype: {tensor.dtype}, requires_grad: {requires_grad}")
            
            return (tensor, info_message, batch_size)

        except Exception as e:
            error_msg = f"Error generating tensor: {str(e)}"
            return (torch.empty(0), error_msg, 0)


import torch
from torch.autograd import grad, functional
import ast
import operator as op


class NntTensorOperations:
    """
    Node for performing tensor operations including custom function evaluation
    and automatic differentiation with support for multiple tensors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "operation": (TENSOR_OPERATIONS, {
                    "default": "add_tensors"
                }),
                "tensor_a": ("TENSOR",),
            },
            "optional": {
                "tensor_b": ("TENSOR",),
                "scalar_value": ("FLOAT", {"default": 1.0}),
                "custom_expression": ("STRING", {"default": "tensor_a * 2"}),
                "grad_tensor": (["tensor_a", "tensor_b"], {"default": "tensor_a"}),
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "STRING")
    RETURN_NAMES = ("result_tensor", "gradient_tensor", "info_message")
    FUNCTION = "perform_operation"
    OUTPUT_NODE = True
    CATEGORY = "NNT Neural Network Toolkit/Tensors"

    def evaluate_custom_function(self, expression_str, tensor_a, tensor_b=None):
        """
        Evaluates a custom function expression using tensor_a and tensor_b
        """
        if not isinstance(expression_str, str):
            raise ValueError(f"Expression must be a string, got {type(expression_str)}")
            
        # Clean the expression string
        expression_str = expression_str.strip().replace(' ', '')
        
        # Ensure input tensors are set up for gradient computation
        if not tensor_a.requires_grad:
            tensor_a.requires_grad_(True)
        if tensor_b is not None and not tensor_b.requires_grad:
            tensor_b.requires_grad_(True)
        
        try:
            if expression_str == "tensor_a*2":
                return torch.mul(tensor_a, 2.0)
                
            if "**" in expression_str:
                base, exponent = expression_str.split("**")
                if base == "tensor_a":
                    return torch.pow(tensor_a, float(exponent))
            
            # Other operations using eval
            locals_dict = {
                'tensor_a': tensor_a,
                'tensor_b': tensor_b if tensor_b is not None else tensor_a,
                'torch': torch
            }
            
            result = eval(expression_str, {"__builtins__": None}, locals_dict)
            return result
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{expression_str}': {str(e)}")
    """
    def compute_gradient(self, result_tensor, wrt_tensor):

        if not isinstance(result_tensor, torch.Tensor):
            raise ValueError("Result must be a tensor for gradient computation")
            
        try:
            # Create scalar result
            scalar_result = result_tensor.sum()
            
            # Compute gradient
            grad = torch.autograd.grad(scalar_result, wrt_tensor, create_graph=True)[0]
            return grad
            
        except Exception as e:
            raise ValueError(f"Error computing gradient: {str(e)}")
    """        

    def prepare_tensor_for_grad(self, tensor):
        
        if tensor is None:
            return None
        if not tensor.is_floating_point():
            tensor = tensor.float()
        return tensor.detach().clone().requires_grad_(True)

        
    def compute_gradient(self, result_tensor, wrt_tensor):
        """
        Compute the gradient of result_tensor with respect to wrt_tensor.
        """
        if not isinstance(result_tensor, torch.Tensor):
            raise ValueError("Result must be a tensor for gradient computation.")
        
        # Ensure result_tensor is part of the computation graph
        if result_tensor.grad_fn is None:
            result_tensor = result_tensor * 1.0  # Ensure it's connected to the graph

        if not result_tensor.requires_grad:
            result_tensor.requires_grad_(True)

        # Ensure wrt_tensor requires grad
        wrt_tensor = wrt_tensor.float()
        if not wrt_tensor.requires_grad:
            wrt_tensor.requires_grad_(True)

        try:
            # Reduce result_tensor to a scalar if necessary
            if result_tensor.numel() > 1:
                result_scalar = result_tensor.sum()
            else:
                result_scalar = result_tensor

            # Zero gradients before computation
            if wrt_tensor.grad is not None:
                wrt_tensor.grad.zero_()

            # Compute gradient
            result_scalar.backward(retain_graph=True)

            # Return gradient
            if wrt_tensor.grad is None:
                raise ValueError("No gradient was computed.")
            return wrt_tensor.grad.clone()

        except Exception as e:
            raise ValueError(f"Error computing gradient: {str(e)}")

        
    def compute_jacobian(self, tensor, create_graph=False, retain_graph=False):
        """Compute Jacobian matrix"""
        tensor = self.prepare_tensor_for_grad(tensor)
        tensor_flat = tensor.reshape(-1)
        jacobian = torch.zeros(tensor_flat.shape[0], tensor_flat.shape[0])
        
        for i in range(tensor_flat.shape[0]):
            gradient = grad(tensor_flat[i], tensor, create_graph=create_graph, 
                          retain_graph=True if i < tensor_flat.shape[0]-1 or retain_graph else False)[0]
            jacobian[i] = gradient.reshape(-1)
            
        return jacobian.reshape(*tensor.shape, *tensor.shape)

    def compute_hessian(self, tensor, create_graph=False, retain_graph=False):
        """Compute Hessian matrix"""
        tensor = self.prepare_tensor_for_grad(tensor)
        gradient = self.compute_gradient(tensor, tensor)
        
        hessian_rows = []
        gradient_flat = gradient.reshape(-1)
        
        for i in range(gradient_flat.shape[0]):
            hessian_row = grad(gradient_flat[i], tensor, create_graph=create_graph,
                             retain_graph=True if i < gradient_flat.shape[0]-1 or retain_graph else False)[0]
            hessian_rows.append(hessian_row.reshape(-1))
            
        return torch.stack(hessian_rows).reshape(*tensor.shape, *tensor.shape)

    def perform_operation(self, operation, tensor_a, tensor_b=None, scalar_value=1.0,
                     custom_expression="tensor_a * 2", grad_tensor="tensor_a"):
        try:
            gradient_tensor = None
            
            if operation == "add_tensors":
                if tensor_b is None:
                    raise ValueError("tensor_b is required for addition.")
                result_tensor = tensor_a + tensor_b
                info_message = "Added tensor_a and tensor_b."
                
            elif operation == "subtract_tensors":
                if tensor_b is None:
                    raise ValueError("tensor_b is required for subtraction.")
                result_tensor = tensor_a - tensor_b
                info_message = "Subtracted tensor_b from tensor_a."
                
            elif operation == "multiply_tensors_elementwise":
                if tensor_b is None:
                    raise ValueError("tensor_b is required for element-wise multiplication.")
                result_tensor = tensor_a * tensor_b
                info_message = "Performed element-wise multiplication of tensor_a and tensor_b."
                
            elif operation == "matrix_multiply_tensors":
                if tensor_b is None:
                    raise ValueError("tensor_b is required for matrix multiplication.")
                result_tensor = torch.matmul(tensor_a, tensor_b)
                info_message = "Performed matrix multiplication of tensor_a and tensor_b."
                
            elif operation == "transpose_tensor":
                result_tensor = tensor_a.transpose(-2, -1)
                info_message = "Transposed tensor_a."
                
            elif operation == "inverse_tensor":
                result_tensor = torch.inverse(tensor_a)
                info_message = "Computed inverse of tensor_a."
                
            elif operation == "add_scalar_to_tensor":
                result_tensor = tensor_a + scalar_value
                info_message = f"Added scalar value {scalar_value} to tensor_a."
                
            elif operation == "multiply_tensor_by_scalar":
                result_tensor = tensor_a * scalar_value
                info_message = f"Multiplied tensor_a by scalar value {scalar_value}."
                
            elif operation == "gradient":
                # Ensure tensor_a is prepared for gradient computation
                tensor_a = tensor_a.float()
                tensor_a.requires_grad_(True)

                # Perform a simple operation to create a result_tensor with grad_fn
                result_tensor = tensor_a * 1.0  # Ensures a computation graph exists
                gradient_tensor = self.compute_gradient(result_tensor, tensor_a)
                info_message = "Computed gradient of tensor_a."
                            
            elif operation == "jacobian":
                tensor_a = tensor_a.float()
                tensor_a.requires_grad_(True)
                result_tensor = self.compute_jacobian(tensor_a)
                info_message = "Computed Jacobian matrix of tensor_a."
                
            elif operation == "hessian":
                tensor_a = tensor_a.float()
                tensor_a.requires_grad_(True)
                result_tensor = self.compute_hessian(tensor_a)
                info_message = "Computed Hessian matrix of tensor_a."
                
            elif operation == "gradient_norm":
                tensor_a = tensor_a.float()
                tensor_a.requires_grad_(True)
                gradient = self.compute_gradient(tensor_a, tensor_a)
                result_tensor = torch.norm(gradient)
                info_message = "Computed gradient norm of tensor_a."
                
            elif operation == "custom_function" or operation == "custom_function_with_grad":
                # Convert tensors to float for gradient computation
                tensor_a = tensor_a.float()
                tensor_a.requires_grad_(True)

                # Perform a simple operation to create a result_tensor with grad_fn
                result_tensor = tensor_a * 1.0  # Ensures a computation graph exists
                tensor_a = tensor_a.float()
                if tensor_b is not None:
                    tensor_b = tensor_b.float()
                
                # Ensure requires_grad is True
                tensor_a.requires_grad_(True)
                if tensor_b is not None:
                    tensor_b.requires_grad_(True)
                
                # Evaluate the custom function
                result_tensor = self.evaluate_custom_function(str(custom_expression), tensor_a, tensor_b)
                info_message = f"Evaluated custom function: {custom_expression}"
                
                # Compute gradient if requested
                if operation == "custom_function_with_grad":
                    try:
                        wrt_tensor = tensor_a if grad_tensor == "tensor_a" else tensor_b
                        if wrt_tensor is None:
                            raise ValueError(f"Cannot compute gradient with respect to {grad_tensor}: tensor is None")
                        gradient_tensor = self.compute_gradient(result_tensor, wrt_tensor)
                        info_message += f"\nCalculated gradient with respect to {grad_tensor}"
                    except Exception as e:
                        info_message += f"\nGradient computation failed: {str(e)}"
                        gradient_tensor = torch.tensor(0.0)
            else:
                raise ValueError(f"Unsupported operation: {operation}")

            # If gradient wasn't calculated, return zero tensor
            if gradient_tensor is None:
                gradient_tensor = torch.tensor(0.0)

            return (result_tensor, gradient_tensor, info_message)
                
        except Exception as e:
            error_message = f"Error during tensor operation: {str(e)}"
            return (tensor_a, torch.tensor(0.0), error_message)

class NntTensorSlice:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_element": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**31-1,
                    "step": 1
                }),
                "num_elements": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 2**31-1,
                    "step": 1
                }),
                "flatten": (["True", "False"], {"default": "False"}),
                "reshape": (["True", "False"], {"default": "False"}),
                "shape": ("STRING", {
                    "default": "[1,1,1]",
                    "multiline": False,
                    "placeholder": "Shape as list, e.g. [1,2,3]"
                }),
                "convert_mask": (["True", "False"], {"default": "False"}),
            },
            "optional": {
                "tensor": ("TENSOR", {"forceInput": True}),
                "image": ("IMAGE", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("TENSOR", "IMAGE") 
    FUNCTION = "slice_tensor"
    CATEGORY = "NNT Neural Network Toolkit/Tensors"

    def slice_tensor(self, start_element, num_elements, flatten, reshape, shape, convert_mask, tensor=None, image=None):
        try:
            import torch
            working_tensor = tensor if tensor is not None else image
            if working_tensor is None:
                raise ValueError("No input tensor provided")

            # Convert to float tensor if needed
            if working_tensor.dtype != torch.float32:
                working_tensor = working_tensor.float()

            if flatten == "True":
                working_tensor = working_tensor.reshape(-1)
            
            total_elements = working_tensor.size(0)
            if start_element >= total_elements:
                raise ValueError(f"Start index {start_element} exceeds tensor size {total_elements}")
            
            if start_element + num_elements > total_elements:
                num_elements = total_elements - start_element
            
            result = working_tensor[start_element:start_element + num_elements]
            
            if reshape == "True":
                try:
                    target_shape = eval(shape)
                    if isinstance(target_shape, (list, tuple)):
                        result = result.reshape(target_shape)
                    else:
                        print("Shape must be a list or tuple")
                except Exception as e:
                    print(f"Reshape failed: {str(e)}")

            if convert_mask == "True":
                # First make it float32 if not already
                result = result.float()
                
                result = result.permute(0, 3, 1, 2)  # Convert to [N,3,H,W]
                
            
            return (result, result)

        except Exception as e:
            print(f"Error slicing tensor: {str(e)}")
            return (tensor if tensor is not None else image, tensor if tensor is not None else image)

class NntTorchvisionDatasets:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_name": (["CIFAR10", "CIFAR100", "MNIST", "FashionMNIST"], {
                    "default": "CIFAR10"
                }),
                "split": (["train", "test"], {
                    "default": "train"
                }),
                "data_dir": ("STRING", {
                    "default": "data/torchvision",
                    "multiline": False,
                }),
                "download": (["True", "False"], {
                    "default": "True"
                }),
                "normalize_data": (["True", "False"], {
                    "default": "True"
                }),
                "enable_augmentation": (["True", "False"], {
                    "default": "True"
                }),
                "samples_to_return": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 512,
                    "step": 1
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50000,
                    "step": 1
                })
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "STRING", "INT")
    RETURN_NAMES = ("images", "labels", "dataset_info", "num_classes")
    FUNCTION = "load_dataset"
    OUTPUT_NODE = True
    CATEGORY = "NNT Neural Network Toolkit/Tensors"

    # In NntTorchvisionDatasets, modify the data loading section:

    def load_dataset(self, dataset_name, split, data_dir, download, normalize_data, 
                    enable_augmentation, samples_to_return, start_index):
        import torch
        import torchvision
        import torchvision.transforms as transforms
        
        try:
            with torch.inference_mode(False), torch.set_grad_enabled(True):
                # Define transforms
                transform_list = [transforms.ToTensor()]
                if normalize_data == "True":
                    transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))  # MNIST specific normalization
                
                transform = transforms.Compose(transform_list)
                
                # Load dataset
                dataset = getattr(torchvision.datasets, dataset_name)(
                    root=data_dir,
                    train=(split == "train"),
                    download=(download == "True"),
                    transform=transform
                )
                
                # Get subset of data
                end_index = min(start_index + samples_to_return, len(dataset))
                indices = list(range(start_index, end_index))
                
                # Extract data
                images, labels = [], []
                for idx in indices:
                    img, label = dataset[idx]
                    images.append(img)
                    labels.append(label)
                
                # Stack into tensors
                images = torch.stack(images)
                labels = torch.tensor(labels, dtype=torch.long)
                
                return (images, labels, f"Dataset loaded with shape {images.shape}", len(dataset.classes))
            
        except Exception as e:
            import traceback
            error_msg = f"Error loading dataset: {str(e)}\n{traceback.format_exc()}"
            return (torch.zeros((1, 1, 28, 28)), torch.zeros(1, dtype=torch.long), error_msg, 0)


class NntPlotTensors:
    """
    Node for visualizing relationships between 2-3 tensors with customizable plot types and labels.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "x_tensor": ("TENSOR",),
                "y_tensor": ("TENSOR",),
                "plot_type": ([
                    "line",
                    "scatter", 
                    "line_and_scatter",
                    "connected_scatter"
                ], {
                    "default": "scatter"
                }),
                "x_label": ("STRING", {
                    "default": "X Value",
                    "multiline": False,
                }),
                "y_label": ("STRING", {
                    "default": "Y Value",
                    "multiline": False,
                }),
                "plot_width": ("INT", {
                    "default": 800,
                    "min": 200,
                    "max": 2000,
                    "step": 50
                }),
                "plot_height": ("INT", {
                    "default": 600,
                    "min": 200,
                    "max": 2000,
                    "step": 50
                }),
                "use_lines": (["True", "False"], {
                    "default": "False"
                }),
            },
            "optional": {
                "y_tensor2": ("TENSOR",),
                "y2_label": ("STRING", {
                    "default": "Y2 Value",
                    "multiline": False,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "plot_tensors"
    CATEGORY = "NNT Neural Network Toolkit/Visualization"

    def plot_tensors(self, x_tensor, y_tensor, plot_type, x_label, y_label, 
                    plot_width, plot_height, use_lines, y_tensor2=None, y2_label=None):
        try:
            import torch
            import numpy as np
            
            # Convert tensors to numpy and flatten if needed
            x_data = x_tensor.detach().cpu().numpy().flatten()
            y_data = y_tensor.detach().cpu().numpy().flatten()
            
            if len(x_data) != len(y_data):
                raise ValueError(f"Tensor length mismatch: x={len(x_data)}, y={len(y_data)}")
                
            y2_data = None
            if y_tensor2 is not None:
                y2_data = y_tensor2.detach().cpu().numpy().flatten()
                if len(x_data) != len(y2_data):
                    raise ValueError(f"Tensor length mismatch: x={len(x_data)}, y2={len(y2_data)}")
            
            from PIL import Image
            import io
            import matplotlib.pyplot as plt

            # Create the plot
            plt.figure(figsize=(plot_width/100, plot_height/100), dpi=100)
            
            if plot_type == "scatter":
                plt.scatter(x_data, y_data, label=y_label, alpha=0.6)
                if y2_data is not None:
                    plt.scatter(x_data, y2_data, label=y2_label, alpha=0.6)
                    
            elif plot_type == "line":
                plt.plot(x_data, y_data, label=y_label)
                if y2_data is not None:
                    plt.plot(x_data, y2_data, label=y2_label)
                    
            elif plot_type == "line_and_scatter":
                plt.plot(x_data, y_data, label=y_label)
                plt.scatter(x_data, y_data, alpha=0.4)
                if y2_data is not None:
                    plt.plot(x_data, y2_data, label=y2_label)
                    plt.scatter(x_data, y2_data, alpha=0.4)
                    
            elif plot_type == "connected_scatter":
                plt.scatter(x_data, y_data, label=y_label, alpha=0.6)
                if use_lines == "True":
                    plt.plot(x_data, y_data, alpha=0.3)
                if y2_data is not None:
                    plt.scatter(x_data, y2_data, label=y2_label, alpha=0.6)
                    if use_lines == "True":
                        plt.plot(x_data, y2_data, alpha=0.3)

            plt.xlabel(x_label)
            plt.ylabel(y_label if y2_label is None else "Value")
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Convert plot to tensor
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            
            # Convert to PIL Image and then to tensor
            image = Image.open(buf).convert('RGB')
            image_tensor = torch.tensor(np.array(image).astype(np.float32) / 255.0)
            image_tensor = torch.unsqueeze(image_tensor, 0)
            
            plt.close()
            buf.close()

            return (image_tensor,)

        except Exception as e:
            import traceback
            error_msg = f"Error plotting tensors: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            # Return empty tensor in case of error
            return (torch.zeros((1, 3, plot_height, plot_width), dtype=torch.float32),)

import torch
import numpy as np
from PIL import Image
import traceback

class NntImageToTensor:
    """
    Node for converting an image (ComfyUI IMAGE) to a Torch Tensor 
    with optional resizing, cropping, color conversion, and flattening.
    Handles input images as torch.Tensor with shape [B, H, W, C].
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 4096,
                    "step": 1
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 4096,
                    "step": 1
                }),
                "crop": (["False", "True"], {
                    "default": "False"
                }),
                "color_mode": (["RGB", "Grayscale"], {
                    "default": "RGB"
                }),
                "flatten": (["False", "True"], {
                    "default": "False"
                }),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    FUNCTION = "image_to_tensor"
    CATEGORY = "NNT Neural Network Toolkit/Tensors"

    def image_to_tensor(self, image, width, height, crop, color_mode, flatten):
        """
        Converts an input IMAGE (torch tensor in ComfyUI) into another TENSOR, 
        optionally resizing, cropping, converting color, and flattening.
        Input image is expected to be torch.Tensor with shape [B, H, W, C].
        """
        try:
            # image is a torch.Tensor of shape [B, H, W, C], range [0,1]
            B, H, W, C = image.shape
            # Prepare a list to store processed images
            processed_images = []

            for i in range(B):
                # Get the ith image
                img = image[i]  # shape [H, W, C]
                # Convert to numpy array and scale to 0-255
                np_image = (img.cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(np_image)

                # Color conversion
                if color_mode == "Grayscale":
                    pil_image = pil_image.convert("L")
                else:  # Ensure it's RGB
                    pil_image = pil_image.convert("RGB")

                # Resize the image
                pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)

                # Optional center crop
                if crop == "True":
                    left = (pil_image.width - width) // 2
                    top = (pil_image.height - height) // 2
                    right = left + width
                    bottom = top + height
                    pil_image = pil_image.crop((left, top, right, bottom))

                # Convert back to torch tensor using your method
                image_tensor = torch.tensor(
                    np.array(pil_image).astype(np.float32) / 255.0
                )
                if color_mode == "Grayscale":
                    # Add channel dimension back for grayscale
                    image_tensor = image_tensor.unsqueeze(0)  # shape [1, H, W]
                else:
                    # Reorder dimensions to [C, H, W]
                    image_tensor = image_tensor.permute(2, 0, 1)  # shape [3, H, W]

                # Optional flatten
                if flatten == "True":
                    image_tensor = image_tensor.view(-1)

                # Add to the list of processed images
                processed_images.append(image_tensor)

            # Stack all processed images back into a single tensor
            if flatten == "True":
                # Each image is 1D, stack into [B, N]
                result_tensor = torch.stack(processed_images, dim=0)
            else:
                # Each image is [C, H, W], stack into [B, C, H, W]
                result_tensor = torch.stack(processed_images, dim=0)

            return (result_tensor,)

        except Exception as e:
            error_msg = f"Error in NntImageToTensor: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            # Return a dummy tensor in case of error
            return (torch.zeros(1),)
