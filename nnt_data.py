import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import ast
import os
import folder_paths


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

TORCHVISION_DATASETS = [
    "CIFAR10",
    "CIFAR100", 
    "MNIST",
    "FashionMNIST",
    "EMNIST",
    "SVHN",
    "STL10",
    "ImageNet",
    "LSUN",
    "CelebA"
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

# Data Loading Node
# In nnt_data.py

class NntTorchvisionDataLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_name": (TORCHVISION_DATASETS, {
                    "default": "MNIST"
                }),
                "split": (["train", "test"], {
                    "default": "train"
                }),
                "data_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Leave empty for default path"
                }),
                "download": (["True", "False"], {
                    "default": "True"
                }),
                "normalize_data": (["True", "False"], {
                    "default": "True"
                }),
                "enable_augmentation": (["True", "False"], {
                    "default": "False"
                }),
                "samples_to_return": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 50000,
                    "step": 1
                }),
                "start_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 50000,
                    "step": 1
                }),
                "use_cache": (["True", "False"], {
                    "default": "True"
                })
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "STRING", "INT")
    RETURN_NAMES = ("images", "labels", "dataset_info", "num_classes")
    FUNCTION = "load_dataset"
    OUTPUT_NODE = True
    CATEGORY = "NNT Neural Network Toolkit/Data Loading"

    @staticmethod
    def get_default_data_path():
        """Get default path for dataset storage"""
        import os
        import folder_paths

        # Register path if not already registered
        if not "torchvision_datasets" in folder_paths.folder_names_and_paths:
            dataset_path = os.path.join(folder_paths.models_dir, "torchvision_datasets")
            folder_paths.add_model_folder_path("torchvision_datasets", dataset_path)
            return dataset_path
        
        # If already registered, get the base path directly from models_dir
        return os.path.join(folder_paths.models_dir, "torchvision_datasets")

    def load_dataset(self, dataset_name, split, data_dir, download, normalize_data, 
                    enable_augmentation, samples_to_return, start_index, use_cache):
        try:
            import torch
            import torchvision
            import torchvision.transforms as transforms
            import os

            # If data_dir is empty, use default path
            if not data_dir:
                data_dir = self.get_default_data_path()
            
            # Create dataset specific directory
            dataset_dir = os.path.join(data_dir, dataset_name.lower())
            os.makedirs(dataset_dir, exist_ok=True)

            # Define transforms
            transform_list = [transforms.ToTensor()]
            if normalize_data == "True":
                transform_list.append(transforms.Normalize((0.5,), (0.5,)))
            
            if enable_augmentation == "True":
                transform_list.extend([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                ])
            
            transform = transforms.Compose(transform_list)

            # Get dataset class
            dataset_class = getattr(torchvision.datasets, dataset_name)
            
            # Load dataset
            dataset = dataset_class(
                root=dataset_dir,
                train=(split == "train"),
                download=(download == "True"),
                transform=transform
            )

            # Select samples
            end_index = min(start_index + samples_to_return, len(dataset))
            images, labels = [], []
            
            for idx in range(start_index, end_index):
                img, label = dataset[idx]
                images.append(img)
                labels.append(label)

            # Stack into tensors
            images = torch.stack(images)
            labels = torch.tensor(labels, dtype=torch.long)

            # Create info message
            info_msg = (
                f"Dataset: {dataset_name}\n"
                f"Split: {split}\n"
                f"Samples: {len(images)} of {len(dataset)}\n"
                f"Image shape: {list(images.shape)}\n"
                f"Directory: {dataset_dir}\n"
                f"Normalization: {normalize_data}\n"
                f"Augmentation: {enable_augmentation}"
            )

            return (images, labels, info_msg, len(dataset.classes))

        except Exception as e:
            error_msg = f"Error loading dataset: {str(e)}"
            return (torch.empty(0), torch.empty(0), error_msg, 0)


class NntFileLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data_source": (DATA_SOURCES, {
                    "default": "numpy"
                }),
                "data_dir": ("STRING", {
                    "default": cls.get_default_data_path(),
                    "multiline": False,
                    "placeholder": "Leave empty for default path"
                }),
                "file_pattern": ("STRING", {
                    "default": "*.npy",
                    "multiline": False,
                }),
                "data_type": (list(TENSOR_DTYPES.keys()), {
                    "default": "float32"
                }),
                # Data processing options
                "normalize": (["True", "False"], {
                    "default": "True"
                }),
                "normalize_range": (NORMALIZE_RANGES, {
                    "default": "0-1"
                }),
                "batch_first": (["True", "False"], {
                    "default": "True"
                }),
                "shuffle": (["True", "False"], {
                    "default": "False"
                }),
                # Image specific options
                "image_channels": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 4,
                    "step": 1
                }),
                "image_size": ("INT", {
                    "default": 224,
                    "min": 16,
                    "max": 2048,
                    "step": 1
                }),
                "image_interpolation": (IMAGE_INTERPOLATION_MODES, {
                    "default": "bilinear"
                }),
                "use_cache": (["True", "False"], {
                    "default": "True"
                })
            },
            "optional": {
                "paired_data_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "paired_file_pattern": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "STRING", "INT")
    RETURN_NAMES = ("data", "paired_data", "info_message", "batch_size")
    FUNCTION = "load_data"
    OUTPUT_NODE = True
    CATEGORY = "NNT Neural Network Toolkit/Data Loading"

    @staticmethod
    def get_default_data_path():
        """Get default path for dataset storage"""
        import os
        import folder_paths

        # Register path if not already registered
        if not "nnt_datasets" in folder_paths.folder_names_and_paths:
            dataset_path = os.path.join(folder_paths.models_dir, "nnt_datasets")
            folder_paths.add_model_folder_path("nnt_datasets", dataset_path)
            return dataset_path
        
        # If already registered, get the base path directly from models_dir
        return os.path.join(folder_paths.models_dir, "nnt_datasets")

    def load_data(self, data_source, data_dir, file_pattern, data_type, normalize, 
                normalize_range, batch_first, shuffle, image_channels, image_size,
                image_interpolation, use_cache, paired_data_dir="", paired_file_pattern=""):
        try:
            import torch
            from torch.utils.data import Dataset, DataLoader, TensorDataset
            import numpy as np
            import os
            import hashlib
            
            dtype_map = {
                'float32': torch.float32,
                'float64': torch.float64,
                'int32': torch.int32,
                'int64': torch.int64,
                'uint8': torch.uint8
            }
            torch_dtype = dtype_map[data_type]
            
            # If data_dir is empty, use default path
            if not data_dir:
                data_dir = self.get_default_data_path()
            
            # Generate cache key if caching is enabled
            cache_key = None
            if use_cache == "True":
                cache_params = f"{data_source}_{data_dir}_{file_pattern}_{data_type}_{normalize}_{normalize_range}"
                cache_key = hashlib.md5(cache_params.encode()).hexdigest()
                
                # Check if data is already in cache
                if cache_key in self._data_cache:
                    tensor, paired_tensor, info_message = self._data_cache[cache_key]
                    return tensor, paired_tensor, f"Loaded from cache: {info_message}", tensor.size(0)

            # Initialize outputs
            tensor = torch.empty(0)
            paired_tensor = torch.empty(0)
            info_message = ""

            # Load data based on source
            if data_source == "python_pickle":
                tensor, paired_tensor, info_message = self._load_pickle_data(
                    data_dir, file_pattern, torch_dtype, normalize, normalize_range
                )
            elif data_source in ["txt", "numpy"]:
                tensor, info_message = self._load_array_data(
                    data_source, data_dir, file_pattern, torch_dtype,
                    normalize, normalize_range, batch_first
                )
                paired_tensor = torch.empty(0)
            elif data_source == "image_folder":
                tensor, info_message = self._load_images(
                    data_dir, file_pattern, image_channels, image_size,
                    image_interpolation, normalize, normalize_range
                )
                paired_tensor = torch.empty(0)
            elif data_source == "image_text_pairs":
                if not paired_data_dir or not paired_file_pattern:
                    raise ValueError("Paired directory path and file pattern required for image-text pairs")
                tensor, paired_tensor, info_message = self._load_image_text_pairs(
                    data_dir, paired_data_dir, file_pattern, paired_file_pattern,
                    image_channels, image_size
                )
            elif data_source == "text_text_pairs":
                if not paired_data_dir or not paired_file_pattern:
                    raise ValueError("Paired directory path and file pattern required for text-text pairs")
                tensor, paired_tensor, info_message = self._load_text_text_pairs(
                    data_dir, paired_data_dir, file_pattern, paired_file_pattern
                )

            # Create dataset and dataloader if shuffling is needed
            if shuffle == "True":
                # Create appropriate dataset based on whether we have paired data
                if paired_tensor.nelement() > 0:
                    dataset = TensorDataset(tensor, paired_tensor)
                else:
                    dataset = TensorDataset(tensor)

                # Create DataLoader
                dataloader = DataLoader(
                    dataset,
                    batch_size=len(dataset),  # Load all data at once
                    shuffle=True
                )

                # Load the data using the DataLoader
                batch = next(iter(dataloader))
                if paired_tensor.nelement() > 0:
                    tensor, paired_tensor = batch
                else:
                    [tensor] = batch

                info_message += "\nData shuffled"

            # Cache the results if requested
            if use_cache == "True":
                self._data_cache[cache_key] = (tensor, paired_tensor, info_message)
                info_message += "\nData cached for future use"

            batch_size = tensor.size(0) if tensor.dim() > 0 else 0
            tensor = tensor.float().requires_grad_(True)  # Make sure tensor has gradients enabled
            if paired_tensor.nelement() > 0:
                paired_tensor = paired_tensor.long()  # Labels should be long type but don't need gradients

            return tensor, paired_tensor, info_message, batch_size

        except Exception as e:
            error_msg = f"Error loading data: {str(e)}"
            return torch.empty(0), torch.empty(0), error_msg, 0

    # Include all the helper methods from the original loader
    def _normalize_tensor(self, tensor, normalize_range):
        """Helper function to normalize a tensor to a specified range."""
        if normalize_range == "0-1":
            min_val = tensor.min()
            max_val = tensor.max()
            if max_val > min_val:
                tensor = (tensor - min_val) / (max_val - min_val)
        elif normalize_range == "-1-1":
            min_val = tensor.min()
            max_val = tensor.max()
            if max_val > min_val:
                tensor = 2.0 * (tensor - min_val) / (max_val - min_val) - 1.0
        elif normalize_range == "standardize":
            mean = tensor.mean()
            std = tensor.std()
            if std > 0:
                tensor = (tensor - mean) / std
        return tensor


    def _load_pickle_data(self, folder_path, file_pattern, dtype, normalize, normalize_range):
        import torch
        import numpy as np
        import os
        import glob
        import pickle
        
        try:
            path_pattern = os.path.join(folder_path, file_pattern)
            files = glob.glob(path_pattern)
            if not files:
                raise ValueError(f"No files found matching pattern: {path_pattern}")

            data_list = []
            label_list = []
            
            for file_path in files:
                with open(file_path, 'rb') as fo:
                    data_dict = pickle.load(fo, encoding='bytes')
# Extract data and labels from the dictionary
                    data = data_dict.get(b'data')
                    if data is None:
                        raise ValueError(f"No data found in {file_path}")
                        
                    labels = data_dict.get(b'labels') or data_dict.get(b'fine_labels')
                    if labels is None:
                        raise ValueError(f"No labels found in {file_path}")
                        
                    data_list.append(data)
                    label_list.extend(labels)

            # Concatenate data and labels
            data_array = np.concatenate(data_list, axis=0)
            labels_array = np.array(label_list)

            # Convert to tensors
            tensor = torch.tensor(data_array, dtype=dtype, requires_grad=True)
            labels_tensor = torch.tensor(labels_array, dtype=torch.long)

            # Normalize if requested
            if normalize == "True":
                tensor = self._normalize_tensor(tensor, normalize_range)

            info_message = f"Loaded {len(files)} pickle files with {len(label_list)} samples"
            return tensor, labels_tensor, info_message

        except Exception as e:
            raise RuntimeError(f"Error loading pickle data: {str(e)}")

    def _load_array_data(self, data_source, folder_path, file_pattern, dtype,
                        normalize, normalize_range, batch_first):
        import torch
        import numpy as np
        import os
        import glob
        
        try:
            path_pattern = os.path.join(folder_path, file_pattern)
            files = glob.glob(path_pattern)
            if not files:
                raise ValueError(f"No files found matching pattern: {path_pattern}")

            data_list = []
            for file_path in files:
                if data_source == "txt":
                    data = np.loadtxt(file_path)
                else:  # numpy
                    data = np.load(file_path)
                data_list.append(data)

            # Stack all arrays
            if len(data_list) == 1:
                data_array = data_list[0]
            else:
                try:
                    data_array = np.stack(data_list, axis=0)
                except ValueError:
                    # If arrays have different shapes, try vstack
                    data_array = np.vstack(data_list)

            # Convert to tensor
            tensor = torch.tensor(data_array, dtype=dtype)

            # Normalize if requested
            if normalize == "True":
                tensor = self._normalize_tensor(tensor, normalize_range)

            # Adjust batch dimension if needed
            if not batch_first == "True" and tensor.dim() > 1:
                tensor = tensor.permute(*range(1, tensor.dim()), 0)

            info_message = f"Loaded {len(files)} {data_source} files"
            return tensor, info_message

        except Exception as e:
            raise RuntimeError(f"Error loading array data: {str(e)}")

    def _load_images(self, folder_path, file_pattern, channels, size,
                    interpolation, normalize, normalize_range):
        import torch
        import numpy as np
        import os
        import glob
        from PIL import Image
        
        try:
            path_pattern = os.path.join(folder_path, file_pattern)
            files = glob.glob(path_pattern)
            if not files:
                raise ValueError(f"No images found matching pattern: {path_pattern}")

            interpolation_modes = {
                "nearest": Image.NEAREST,
                "bilinear": Image.BILINEAR,
                "bicubic": Image.BICUBIC
            }
            interp_mode = interpolation_modes.get(interpolation, Image.BILINEAR)

            data_list = []
            for file_path in files:
                try:
                    img = Image.open(file_path)
                    
                    # Convert image mode based on channels
                    if channels == 1:
                        img = img.convert('L')
                    elif channels == 3:
                        img = img.convert('RGB')
                    elif channels == 4:
                        img = img.convert('RGBA')
                    else:
                        raise ValueError(f"Unsupported number of channels: {channels}")

                    # Resize image
                    img = img.resize((size, size), interp_mode)
                    
                    # Convert to numpy array
                    img_array = np.array(img).astype(np.float32)
                    
                    # Add channel dimension for grayscale
                    if channels == 1:
                        img_array = img_array[..., np.newaxis]
                    
                    data_list.append(img_array)

                except Exception as e:
                    raise RuntimeError(f"Error processing image {file_path}: {str(e)}")

            # Stack all images
            data_array = np.stack(data_list, axis=0)
            
            # Normalize to [0, 1] initially
            if data_array.max() > 1.0:
                data_array = data_array / 255.0

            # Convert to tensor and arrange dimensions to (N, C, H, W)
            tensor = torch.from_numpy(data_array)
            if channels == 1:
                tensor = tensor.permute(0, 3, 1, 2)  # (N, H, W, 1) -> (N, 1, H, W)
            else:
                tensor = tensor.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

            # Apply additional normalization if requested
            if normalize == "True":
                tensor = self._normalize_tensor(tensor, normalize_range)

            info_message = f"Loaded {len(files)} images with shape {tuple(tensor.shape)}"
            return tensor, info_message

        except Exception as e:
            raise RuntimeError(f"Error loading images: {str(e)}")

    def _load_image_text_pairs(self, image_folder, text_folder, image_pattern,
                              text_pattern, channels, size, vocab_size,
                              sequence_length, tokenizer):
        import torch
        import numpy as np
        import os
        import glob
        from PIL import Image
        from collections import Counter
        
        try:
            # Get matching files
            image_files = glob.glob(os.path.join(image_folder, image_pattern))
            text_files = glob.glob(os.path.join(text_folder, text_pattern))

            image_basenames = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
            text_basenames = {os.path.splitext(os.path.basename(f))[0]: f for f in text_files}

            common_basenames = sorted(set(image_basenames.keys()) & set(text_basenames.keys()))
            if not common_basenames:
                raise ValueError("No matching image-text pairs found")

            # Build vocabulary first
            word_counts = Counter()
            for bn in common_basenames:
                with open(text_basenames[bn], 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if tokenizer == "basic":
                        tokens = text.split()
                    elif tokenizer == "wordpiece":
                        tokens = []
                        for word in text.split():
                            if len(word) > 1:
                                tokens.extend([word[0]] + [f"##{c}" for c in word[1:]])
                            else:
                                tokens.append(word)
                    elif tokenizer == "bpe":
                        tokens = [c for c in text]  # Character-level as simplification
                    word_counts.update(tokens)

            # Create vocabulary
            vocab = {"<pad>": 0, "<unk>": 1}
            for word, _ in word_counts.most_common(vocab_size - 2):  # -2 for pad and unk
                if word not in vocab:
                    vocab[word] = len(vocab)

            # Process pairs
            image_tensors = []
            text_tensors = []

            for bn in common_basenames:
                # Process image
                img = Image.open(image_basenames[bn])
                img = img.convert('RGB' if channels == 3 else 'L')
                img = img.resize((size, size))
                img_array = np.array(img).astype(np.float32) / 255.0
                if channels == 1:
                    img_array = img_array[..., np.newaxis]

                # Process text
                with open(text_basenames[bn], 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    if tokenizer == "basic":
                        tokens = text.split()
                    elif tokenizer == "wordpiece":
                        tokens = []
                        for word in text.split():
                            if len(word) > 1:
                                tokens.extend([word[0]] + [f"##{c}" for c in word[1:]])
                            else:
                                tokens.append(word)
                    elif tokenizer == "bpe":
                        tokens = [c for c in text]

                # Convert tokens to indices
                indices = [vocab.get(token, vocab["<unk>"]) for token in tokens[:sequence_length]]
                indices = indices + [vocab["<pad>"]] * (sequence_length - len(indices))

                image_tensors.append(torch.from_numpy(img_array))
                text_tensors.append(torch.tensor(indices, dtype=torch.long))

            # Stack all tensors
            image_tensor = torch.stack(image_tensors).permute(0, 3, 1, 2)
            text_tensor = torch.stack(text_tensors)

            info_message = (f"Loaded {len(common_basenames)} image-text pairs\n"
                          f"Vocabulary size: {len(vocab)}")
            return image_tensor, text_tensor, info_message

        except Exception as e:
            raise RuntimeError(f"Error loading image-text pairs: {str(e)}")

    def _load_text_text_pairs(self, text_folder_1, text_folder_2,
                             pattern_1, pattern_2, vocab_size,
                             sequence_length, tokenizer):
        import torch
        import os
        import glob
        from collections import Counter
        
        try:
            # Get matching files
            files_1 = glob.glob(os.path.join(text_folder_1, pattern_1))
            files_2 = glob.glob(os.path.join(text_folder_2, pattern_2))

            basenames_1 = {os.path.splitext(os.path.basename(f))[0]: f for f in files_1}
            basenames_2 = {os.path.splitext(os.path.basename(f))[0]: f for f in files_2}

            common_basenames = sorted(set(basenames_1.keys()) & set(basenames_2.keys()))
            if not common_basenames:
                raise ValueError("No matching text pairs found")

            # Build shared vocabulary
            word_counts = Counter()
            for bn in common_basenames:
                for file_path in [basenames_1[bn], basenames_2[bn]]:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if tokenizer == "basic":
                            tokens = text.split()
                        elif tokenizer == "wordpiece":
                            tokens = []
                            for word in text.split():
                                if len(word) > 1:
                                    tokens.extend([word[0]] + [f"##{c}" for c in word[1:]])
                                else:
                                    tokens.append(word)
                        elif tokenizer == "bpe":
                            tokens = [c for c in text]
                        word_counts.update(tokens)

            # Create vocabulary
            vocab = {"<pad>": 0, "<unk>": 1}
            for word, _ in word_counts.most_common(vocab_size - 2):
                if word not in vocab:
                    vocab[word] = len(vocab)

            # Process pairs
            tensors_1 = []
            tensors_2 = []

            for bn in common_basenames:
                for file_path, tensor_list in [(basenames_1[bn], tensors_1),
                                             (basenames_2[bn], tensors_2)]:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if tokenizer == "basic":
                            tokens = text.split()
                        elif tokenizer == "wordpiece":
                            tokens = []
                            for word in text.split():
                                if len(word) > 1:
                                    tokens.extend([word[0]] + [f"##{c}" for c in word[1:]])
                                else:
                                    tokens.append(word)
                        elif tokenizer == "bpe":
                            tokens = [c for c in text]

                    indices = [vocab.get(token, vocab["<unk>"]) for token in tokens[:sequence_length]]
                    indices = indices + [vocab["<pad>"]] * (sequence_length - len(indices))
                    tensor_list.append(torch.tensor(indices, dtype=torch.long))

            # Stack all tensors
            tensor_1 = torch.stack(tensors_1)
            tensor_2 = torch.stack(tensors_2)

            info_message = (f"Loaded {len(common_basenames)} text pairs\n"
                          f"Vocabulary size: {len(vocab)}")
            return tensor_1, tensor_2, info_message

        except Exception as e:
            raise RuntimeError(f"Error loading text pairs: {str(e)}")


class NntTimeSeriesDataLoader:
    """
    Node for loading common time series datasets from statsmodels and other sources,
    with built-in preprocessing and analysis capabilities.
    """
    
    import torch
    import numpy as np
    from statsmodels.datasets import get_rdataset
    import pandas as pd
    from scipy import stats
    import os

    DATASETS = [
        "airline_passengers",
        "sunspots",
        "mortality_rates",
        "livestock",
        "electricity_consumption",
        "gas_prices",
        "temperature",
        "unemployment",
        "stock_returns",
        "exchange_rates",
        "co2_levels"
    ]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (cls.DATASETS, {
                    "default": "airline_passengers"
                }),
                "start_date": ("STRING", {
                    "default": "1949-01",
                    "multiline": False,
                    "placeholder": "YYYY-MM or YYYY-MM-DD"
                }),
                "end_date": ("STRING", {
                    "default": "1960-12",
                    "multiline": False,
                    "placeholder": "YYYY-MM or YYYY-MM-DD"
                }),
                "frequency": (["M", "D", "W", "Y", "Q", "H"], {
                    "default": "M"
                }),
                "preprocessing": (["None", "standardize", "normalize", "log", "difference", "box-cox"], {
                    "default": "None"
                }),
                "fill_missing": (["forward", "backward", "linear", "none"], {
                    "default": "forward"
                }),
                "return_type": (["single_series", "multi_series", "features_targets"], {
                    "default": "single_series"
                }),
                "sequence_length": ("INT", {
                    "default": 60,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "prediction_horizon": ("INT", {
                    "default": 12,
                    "min": 1,
                    "max": 100,
                    "step": 1
                }),
            },
            "optional": {
                "custom_filepath": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Path to custom CSV file"
                }),
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "STRING", "DICT")
    RETURN_NAMES = ("time_series_data", "features_targets", "dataset_info", "dataset_stats")
    FUNCTION = "load_time_series"
    OUTPUT_NODE = True
    CATEGORY = "NNT Neural Network Toolkit/Data Loading"

    @staticmethod 
    def get_default_data_path():
        """Get default path for dataset storage"""
        import os
        import folder_paths

        if not "timeseries_datasets" in folder_paths.folder_names_and_paths:
            dataset_path = os.path.join(folder_paths.models_dir, "timeseries_datasets")
            folder_paths.add_model_folder_path("timeseries_datasets", dataset_path)
            return dataset_path

        return os.path.join(folder_paths.models_dir, "timeseries_datasets")

    def load_time_series(self, dataset, start_date, end_date, frequency, preprocessing,
                        fill_missing, return_type, sequence_length, prediction_horizon,
                        cache_dir="", custom_filepath=""):
        try:
            
            import pandas as pd
            # Handle data directory
            if not cache_dir:
                cache_dir = self.get_default_data_path()
                
            # Create dataset specific directory
            dataset_dir = os.path.join(cache_dir, dataset.lower())
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Dictionary to store dataset information and statistics
            dataset_stats = {
                "original_shape": None,
                "processed_shape": None,
                "missing_values": 0,
                "seasonality_test": None,
                "stationarity_test": None,
                "basic_stats": {},
                "frequency": frequency,
                "dataset_dir": dataset_dir
            }

            # Load dataset
            if custom_filepath:
                if not os.path.exists(custom_filepath):
                    raise FileNotFoundError(f"Custom file not found: {custom_filepath}")
                df = pd.read_csv(custom_filepath, parse_dates=True, index_col=0)
            else:
                df = self._load_dataset(dataset)

                # Save loaded dataset to cache directory
                cache_file = os.path.join(dataset_dir, f"{dataset.lower()}_raw.csv")
                df.to_csv(cache_file)
                dataset_stats["cache_file"] = cache_file

            # Store original shape
            dataset_stats["original_shape"] = df.shape
            dataset_stats["missing_values"] = df.isnull().sum().sum()

            # Handle missing values
            if fill_missing != "none":
                df = self._handle_missing_values(df, fill_missing)

            # Preprocess data
            if preprocessing != "None":
                df = self._preprocess_data(df, preprocessing)
                
                # Save preprocessed data
                preprocessed_file = os.path.join(dataset_dir, f"{dataset.lower()}_preprocessed.csv")
                df.to_csv(preprocessed_file)
                dataset_stats["preprocessed_file"] = preprocessed_file
                
            # Perform statistical tests
            dataset_stats["stationarity_test"] = self._test_stationarity(df)
            dataset_stats["seasonality_test"] = self._test_seasonality(df)
            
            # Calculate basic statistics
            dataset_stats["basic_stats"] = {
                "mean": float(df.mean()),
                "std": float(df.std()),
                "min": float(df.min()),
                "max": float(df.max()),
                "skewness": float(stats.skew(df.values)),
                "kurtosis": float(stats.kurtosis(df.values))
            }

            # Prepare data based on return type
            if return_type == "single_series":
                data_tensor = torch.tensor(df.values, dtype=torch.float32)
                features_tensor = torch.empty(0)  # Empty tensor for unused return
                
                # Save formatted data
                np.save(os.path.join(dataset_dir, f"{dataset.lower()}_single_series.npy"), 
                    data_tensor.numpy())
            
            elif return_type == "multi_series":
                # Create multiple series with different lags
                series_list = []
                for i in range(sequence_length):
                    series_list.append(df.shift(i))
                data = pd.concat(series_list, axis=1).dropna()
                data_tensor = torch.tensor(data.values, dtype=torch.float32)
                features_tensor = torch.empty(0)
                
                # Save formatted data
                np.save(os.path.join(dataset_dir, f"{dataset.lower()}_multi_series.npy"), 
                    data_tensor.numpy())

            else:  # features_targets
                # Create sequences for feature-target pairs
                features, targets = self._create_sequences(
                    df, sequence_length, prediction_horizon
                )
                data_tensor = torch.tensor(features, dtype=torch.float32)
                features_tensor = torch.tensor(targets, dtype=torch.float32)
                
                # Save formatted data
                np.save(os.path.join(dataset_dir, f"{dataset.lower()}_features.npy"), 
                    data_tensor.numpy())
                np.save(os.path.join(dataset_dir, f"{dataset.lower()}_targets.npy"), 
                    features_tensor.numpy())

            # Store processed shape
            dataset_stats["processed_shape"] = data_tensor.shape

            # Create dataset info string
            dataset_info = self._create_dataset_info(
                dataset, df, dataset_stats, preprocessing, return_type
            )

            # Add additional metadata
            dataset_stats.update({
                "start_date": str(df.index[0]),
                "end_date": str(df.index[-1]),
                "return_type": return_type,
                "sequence_length": sequence_length,
                "prediction_horizon": prediction_horizon,
                "preprocessing": preprocessing,
                "fill_missing": fill_missing
            })

            return (data_tensor, features_tensor, dataset_info, dataset_stats)

        except Exception as e:
            import traceback
            error_msg = f"Error loading time series data: {str(e)}\n{traceback.format_exc()}"
            return (torch.empty(0), torch.empty(0), error_msg, {})

    def _load_dataset(self, dataset):
        """Helper method to load different datasets"""
        import pandas as pd
        from statsmodels.datasets import get_rdataset
        
        if dataset == "airline_passengers":
            data = get_rdataset("AirPassengers", "datasets").data
            return pd.Series(data["value"].values, index=pd.date_range(start="1949-01", end="1960-12", freq="M"))
        
        elif dataset == "sunspots":
            data = get_rdataset("sunspots", "datasets").data
            return pd.Series(data["value"].values, index=pd.date_range(start="1749-01", end="1983-12", freq="Y"))
            
        # Add more datasets as needed...
        
        raise ValueError(f"Dataset {dataset} not found")

    def _handle_missing_values(self, df, method):
        """Helper method to handle missing values"""
        if method == "forward":
            return df.fillna(method="ffill")
        elif method == "backward":
            return df.fillna(method="bfill")
        elif method == "linear":
            return df.interpolate(method="linear")
        return df

    def _preprocess_data(self, df, method):
        """Helper method for data preprocessing"""
        if method == "standardize":
            return (df - df.mean()) / df.std()
        elif method == "normalize":
            return (df - df.min()) / (df.max() - df.min())
        elif method == "log":
            return np.log1p(df)
        elif method == "difference":
            return df.diff().dropna()
        elif method == "box-cox":
            return pd.Series(stats.boxcox(df.values)[0], index=df.index)
        return df

    def _test_stationarity(self, series):
        """Perform Augmented Dickey-Fuller test for stationarity"""
        from statsmodels.tsa.stattools import adfuller
        try:
            result = adfuller(series.values)
            return {
                "test_statistic": float(result[0]),
                "p_value": float(result[1]),
                "is_stationary": result[1] < 0.05
            }
        except:
            return None

    def _test_seasonality(self, series):
        """Test for seasonality using seasonal decompose"""
        from statsmodels.tsa.seasonal import seasonal_decompose
        try:
            result = seasonal_decompose(series, period=12)
            return {
                "seasonal_strength": float(result.seasonal.std() / result.resid.std()),
                "trend_strength": float(result.trend.std() / result.resid.std())
            }
        except:
            return None

    def _create_sequences(self, series, sequence_length, prediction_horizon):
        """Create sequences for feature-target pairs"""
        data = series.values
        X, y = [], []
        
        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            X.append(data[i:(i + sequence_length)])
            y.append(data[(i + sequence_length):(i + sequence_length + prediction_horizon)])
            
        return np.array(X), np.array(y)

    def _create_dataset_info(self, dataset, df, stats, preprocessing, return_type):
        """Create informative dataset summary"""
        info = [
            f"Dataset: {dataset}",
            f"Time Range: {df.index[0]} to {df.index[-1]}",
            f"Frequency: {stats['frequency']}",
            f"Sample Size: {len(df)}",
            f"Preprocessing: {preprocessing}",
            f"Return Type: {return_type}",
            "\nBasic Statistics:",
            f"Mean: {stats['basic_stats']['mean']:.3f}",
            f"Std: {stats['basic_stats']['std']:.3f}",
            f"Skewness: {stats['basic_stats']['skewness']:.3f}",
            f"Kurtosis: {stats['basic_stats']['kurtosis']:.3f}",
        ]

        if stats['stationarity_test']:
            info.extend([
                "\nStationarity Test:",
                f"Test Statistic: {stats['stationarity_test']['test_statistic']:.3f}",
                f"P-value: {stats['stationarity_test']['p_value']:.3f}",
                f"Is Stationary: {stats['stationarity_test']['is_stationary']}"
            ])

        if stats['seasonality_test']:
            info.extend([
                "\nSeasonality Analysis:",
                f"Seasonal Strength: {stats['seasonality_test']['seasonal_strength']:.3f}",
                f"Trend Strength: {stats['seasonality_test']['trend_strength']:.3f}"
            ])

        return "\n".join(info)
    
class NntHuggingFaceDataLoader:
    """Node for loading datasets from HuggingFace Hub."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {
                    "default": "mnist",
                    "multiline": False,
                    "placeholder": "e.g., mnist, cifar10, imdb"
                }),
                "split": (["train", "test", "validation"], {
                    "default": "train"
                }),
                "use_auth_token": (["True", "False"], {
                    "default": "False"
                }),
                "token": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "HuggingFace API token"
                }),
                "cache_dir": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Leave empty for default cache"
                }),
                "num_samples": ("INT", {
                    "default": 1000,
                    "min": 1,
                    "max": 100000,
                    "step": 1
                }),
                "start_idx": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 1
                }),
                "shuffle": (["True", "False"], {
                    "default": "True"
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 99999999,
                    "step": 1
                })
            }
        }

    RETURN_TYPES = ("DATASET", "STRING", "DICT")
    RETURN_NAMES = ("dataset", "info", "metadata")
    FUNCTION = "load_dataset"
    CATEGORY = "NNT Neural Network Toolkit/Data Loading"

    def load_dataset(self, repo_id, split, use_auth_token, token, cache_dir, 
                    num_samples, start_idx, shuffle, seed):
        try:
            from datasets import load_dataset
            import torch
            
            # Handle empty cache_dir
            if not cache_dir.strip():
                cache_dir = None

            # Load dataset
            auth_token = token if use_auth_token == "True" else None
            dataset = load_dataset(repo_id, split=split, token=auth_token, cache_dir=cache_dir)

            # Apply shuffling if requested
            if shuffle == "True":
                dataset = dataset.shuffle(seed=seed)

            # Select subset of data
            end_idx = min(start_idx + num_samples, len(dataset))
            dataset = dataset.select(range(start_idx, end_idx))

            # Create metadata
            metadata = {
                "dataset_info": dataset.info,
                "features": dataset.features,
                "total_samples": len(dataset),
                "selected_samples": end_idx - start_idx,
                "columns": dataset.column_names
            }

            # Create info string
            info = (
                f"Dataset: {repo_id}\n"
                f"Split: {split}\n"
                f"Selected samples: {end_idx - start_idx} / {len(dataset)}\n"
                f"Features: {dataset.column_names}\n"
                f"Cache directory: {cache_dir if cache_dir else 'default'}"
            )

            return (dataset, info, metadata)

        except Exception as e:
            error_msg = f"Error loading dataset: {str(e)}"
            return (None, error_msg, {})

class NntDatasetToImageTensor:
    """Node for converting HuggingFace dataset image columns to tensors with configurable preprocessing."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("DATASET",),
                "image_column": ("STRING", {
                    "default": "image",
                    "multiline": False
                }),
                "target_size": ("INT", {
                    "default": 224,
                    "min": 16,
                    "max": 4096,
                    "step": 16
                }),
                "normalization": (["None", "0-1", "-1-1", "custom"], {
                    "default": "0-1"
                }),
                "custom_mean": ("STRING", {
                    "default": "[0.485, 0.456, 0.406]",
                    "multiline": False,
                    "placeholder": "RGB mean values"
                }),
                "custom_std": ("STRING", {
                    "default": "[0.229, 0.224, 0.225]",
                    "multiline": False,
                    "placeholder": "RGB std values"
                }),
                "num_channels": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 4,
                    "step": 1
                }),
                "interpolation": (["nearest", "bilinear", "bicubic", "lanczos"], {
                    "default": "bilinear"
                }),
                "data_format": (["channels_first", "channels_last"], {
                    "default": "channels_first"
                })
            }
        }

    RETURN_TYPES = ("TENSOR", "STRING")
    RETURN_NAMES = ("image_tensor", "info")
    FUNCTION = "process_images"
    CATEGORY = "NNT Neural Network Toolkit/Data Processing"

    def process_images(self, dataset, image_column, target_size, normalization,
                      custom_mean, custom_std, num_channels, interpolation, data_format):
        try:
            import torch
            import torchvision.transforms as transforms
            from PIL import Image
            import numpy as np
            import ast
            with torch.inference_mode(False), torch.set_grad_enabled(True):
                # Parse custom normalization parameters
                try:
                    mean = ast.literal_eval(custom_mean)
                    std = ast.literal_eval(custom_std)
                except:
                    mean = [0.485, 0.456, 0.406]
                    std = [0.229, 0.224, 0.225]

                # Create transform pipeline
                transform_list = [
                    transforms.Resize((target_size, target_size), 
                        interpolation=getattr(Image, interpolation.upper())),
                    transforms.ToTensor()
                ]

                # Add normalization
                if normalization == "0-1":
                    pass  # ToTensor already normalizes to 0-1
                elif normalization == "-1-1":
                    transform_list.append(transforms.Normalize([0.5] * num_channels, [0.5] * num_channels))
                elif normalization == "custom":
                    transform_list.append(transforms.Normalize(mean[:num_channels], std[:num_channels]))

                transform = transforms.Compose(transform_list)

                # Process images
                images = []
                for example in dataset:
                    img = example[image_column]
                    if isinstance(img, (str, bytes)):
                        img = Image.open(img)
                    img = img.convert('RGB' if num_channels == 3 else 'L')
                    img_tensor = transform(img)
                    images.append(img_tensor)

                # Stack tensors
                image_tensor = torch.stack(images)

                # Adjust format if needed
                if data_format == "channels_last":
                    image_tensor = image_tensor.permute(0, 2, 3, 1)

                info = (
                    f"Processed {len(images)} images\n"
                    f"Output shape: {list(image_tensor.shape)}\n"
                    f"Normalization: {normalization}\n"
                    f"Format: {data_format}"
                )

                return (image_tensor, info)

        except Exception as e:
            error_msg = f"Error processing images: {str(e)}"
            return (torch.empty(0), error_msg)

class NntDatasetToTextTensor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("DATASET",),
                "text_column": ("STRING", {
                    "default": "text",
                    "multiline": False
                }),
                "tokenizer_name": ("STRING", {
                    "default": "bert-base-uncased",
                    "multiline": False
                }),
                "max_length": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 2048,
                    "step": 1
                }),
                # Data collation options
                "use_data_collator": (["True", "False"], {
                    "default": "True"
                }),
                "padding": (["max_length", "longest", "do_not_pad"], {
                    "default": "max_length"
                }),
                "truncation": (["True", "False"], {
                    "default": "True"
                }),
                "add_special_tokens": (["True", "False"], {
                    "default": "True"
                }),
                "return_type": (["input_ids", "attention_mask", "token_type_ids", "all"], {
                    "default": "input_ids"
                }),
                "pad_to_multiple_of": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 128,
                    "step": 1
                }),
                "return_tensors": (["pt", "tf"], {
                    "default": "pt"
                }),
                # Tensor processing options
                "detach_tensor": (["True", "False"], {
                    "default": "True"
                }),
                "requires_grad": (["True", "False"], {
                    "default": "True"
                }),
                "make_clone": (["True", "False"], {
                    "default": "True"
                })
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "DICT", "STRING")
    RETURN_NAMES = ("text_tensor", "attention_mask", "collated_outputs", "info")
    FUNCTION = "process_text"
    CATEGORY = "NNT Neural Network Toolkit/Data Processing"

    def process_text(self, dataset, text_column, tokenizer_name, max_length,
                    use_data_collator, padding, truncation, add_special_tokens, 
                    return_type, pad_to_multiple_of, return_tensors,
                    detach_tensor, requires_grad, make_clone):
        try:
            from transformers import AutoTokenizer, DataCollatorWithPadding
            import torch
            with torch.inference_mode(False), torch.set_grad_enabled(True):

                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

                # Process text
                texts = dataset[text_column]
                
                # Basic tokenization
                encodings = tokenizer(
                    texts,
                    max_length=max_length,
                    padding=False,  # We'll handle padding with collator if needed
                    truncation=truncation == "True",
                    add_special_tokens=add_special_tokens == "True",
                    return_tensors=None  # Don't convert to tensors yet
                )

                # Apply data collation if requested
                if use_data_collator == "True":
                    collator = DataCollatorWithPadding(
                        tokenizer=tokenizer,
                        padding=padding,
                        max_length=max_length,
                        pad_to_multiple_of=pad_to_multiple_of if pad_to_multiple_of > 0 else None,
                        return_tensors=return_tensors
                    )
                    
                    # Collate the encodings
                    collated_outputs = collator([{k: v for k, v in zip(encodings.keys(), enc)} 
                                            for enc in zip(*encodings.values())])
                    
                    # Get main tensor based on return_type
                    if return_type == "all":
                        text_tensor = collated_outputs["input_ids"]
                        attention_mask = collated_outputs["attention_mask"]
                    else:
                        text_tensor = collated_outputs[return_type]
                        attention_mask = torch.empty(0)
                else:
                    # Convert to tensors without collation
                    collated_outputs = {}
                    if return_type == "all":
                        text_tensor = torch.tensor(encodings["input_ids"])
                        attention_mask = torch.tensor(encodings["attention_mask"])
                    else:
                        text_tensor = torch.tensor(encodings[return_type])
                        attention_mask = torch.empty(0)

                # Apply tensor processing options
                def prepare_tensor(tensor):
                    if detach_tensor == "True":
                        tensor = tensor.detach()
                    if make_clone == "True":
                        tensor = tensor.clone()
                    if requires_grad == "True":
                        tensor = tensor.requires_grad_(True)
                    return tensor

                text_tensor = prepare_tensor(text_tensor)
                if attention_mask.numel() > 0:
                    attention_mask = prepare_tensor(attention_mask)

                # Create info message
                info = (
                    f"Processed {len(texts)} texts\n"
                    f"Tokenizer: {tokenizer_name}\n"
                    f"Max length: {max_length}\n"
                    f"Output shape: {list(text_tensor.shape)}\n"
                    f"Data collation: {use_data_collator}\n"
                    f"Padding strategy: {padding if use_data_collator == 'True' else 'none'}\n"
                    f"Tensor properties:\n"
                    f"- Detached: {detach_tensor}\n"
                    f"- Requires grad: {requires_grad}\n"
                    f"- Cloned: {make_clone}\n"
                    f"First row dataset: {dataset[text_column][0]}"
                )

                return (text_tensor, attention_mask, collated_outputs, info)

        except Exception as e:
            error_msg = f"Error processing text: {str(e)}"
            return (torch.empty(0), torch.empty(0), {}, error_msg)


class NntDatasetToTargetTensor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("DATASET",),
                "target_column": ("STRING", {
                    "default": "label",
                    "multiline": False
                }),
                "target_type": (["classification", "regression", "multi_label"], {
                    "default": "classification"
                }),
                "num_classes": ("INT", {
                    "default": 10,
                    "min": 2,
                    "max": 1000,
                    "step": 1
                }),

                "return_tensors": (["pt", "tf"], {
                    "default": "pt"
                }),
                "encoding": (["sparse", "one_hot", "label_smooth"], {
                    "default": "sparse"
                }),
                "label_smoothing": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01
                }),
                # Label mapping options
                "create_label_maps": (["True", "False"], {
                    "default": "True"
                }),
                "custom_label_map": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                    "placeholder": "Optional label mapping dict, e.g., {'NEGATIVE': 0, 'POSITIVE': 1}"
                }),
                # Tensor processing options
                "detach_tensor": (["True", "False"], {
                    "default": "True"
                }),
                "requires_grad": (["True", "False"], {
                    "default": "False"  # Usually False for target tensors
                }),
                "make_clone": (["True", "False"], {
                    "default": "True"
                })
            }
        }

    RETURN_TYPES = ("TENSOR", "STRING", "DICT", "DICT")
    RETURN_NAMES = ("target_tensor", "info", "label_info", "collated_outputs")
    FUNCTION = "process_targets"
    CATEGORY = "NNT Neural Network Toolkit/Data Processing"


class NntDatasetToTargetTensor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("DATASET",),
                "target_column": ("STRING", {
                    "default": "label",
                    "multiline": False
                }),
                "target_type": (["classification", "regression", "multi_label"], {
                    "default": "classification"
                }),
                "num_classes": ("INT", {
                    "default": 10,
                    "min": 2,
                    "max": 1000,
                    "step": 1
                }),
                # Data collation options
                "use_data_collator": (["True", "False"], {
                    "default": "True"
                }),
                "padding": (["max_length", "longest", "do_not_pad"], {
                    "default": "max_length"
                }),
                "pad_to_multiple_of": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 128,
                    "step": 1
                }),
                "return_tensors": (["pt", "tf"], {
                    "default": "pt"
                }),
                "encoding": (["sparse", "one_hot", "label_smooth"], {
                    "default": "sparse"
                }),
                "label_smoothing": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 0.5,
                    "step": 0.01
                }),
                # Label mapping options
                "create_label_maps": (["True", "False"], {
                    "default": "True"
                }),
                "custom_label_map": ("STRING", {
                    "default": "{}",
                    "multiline": True,
                    "placeholder": "Optional label mapping dict, e.g., {'NEGATIVE': 0, 'POSITIVE': 1}"
                }),
                # Tensor processing options
                "detach_tensor": (["True", "False"], {
                    "default": "True"
                }),
                "requires_grad": (["True", "False"], {
                    "default": "False"  # Usually False for target tensors
                }),
                "make_clone": (["True", "False"], {
                    "default": "True"
                })
            }
        }

    RETURN_TYPES = ("TENSOR", "STRING", "DICT", "DICT")
    RETURN_NAMES = ("target_tensor", "info", "label_info", "collated_outputs")
    FUNCTION = "process_targets"
    CATEGORY = "NNT Neural Network Toolkit/Data Processing"

    def process_targets(self, dataset, target_column, target_type, num_classes,
                    use_data_collator, padding, pad_to_multiple_of, return_tensors,
                    encoding, label_smoothing, create_label_maps, custom_label_map,
                    detach_tensor, requires_grad, make_clone):
        try:
            import torch
            from transformers import DataCollatorWithPadding
            import numpy as np
            import ast
            
            with torch.inference_mode(False), torch.set_grad_enabled(True):
                # Get targets and ensure they're in a list
                targets = list(dataset[target_column])
                
                # Convert strings to integers if needed
                if isinstance(targets[0], str):
                    # Process label mappings
                    if create_label_maps == "True":
                        if custom_label_map and custom_label_map != "{}":
                            try:
                                label2id = ast.literal_eval(custom_label_map)
                            except:
                                raise ValueError(f"Invalid label mapping format: {custom_label_map}")
                        else:
                            unique_labels = sorted(set(targets))
                            label2id = {label: idx for idx, label in enumerate(unique_labels)}
                        
                        # Convert string labels to indices
                        targets = [label2id[label] for label in targets]
                        id2label = {v: k for k, v in label2id.items()}
                        num_classes = len(label2id)  # Update num_classes based on mapping
                        
                        label_mappings = {
                            "label2id": label2id,
                            "id2label": id2label,
                            "num_labels": len(label2id)
                        }
                    else:
                        label_mappings = {}
                else:
                    label_mappings = {}
                    if isinstance(targets[0], bool):
                        targets = [int(t) for t in targets]

                # Convert to numpy array first
                targets = np.array(targets)
                
                # Process based on target type and encoding
                if target_type == "classification":
                    if encoding == "sparse":
                        target_tensor = torch.tensor(targets, dtype=torch.long)
                    elif encoding == "one_hot":
                        # Convert to integers first
                        targets_int = torch.tensor(targets, dtype=torch.long)
                        target_tensor = torch.zeros(len(targets), num_classes)
                        target_tensor.scatter_(1, targets_int.unsqueeze(1), 1)
                    else:  # label_smooth
                        targets_int = torch.tensor(targets, dtype=torch.long)
                        target_tensor = torch.zeros(len(targets), num_classes)
                        target_tensor.fill_(label_smoothing / (num_classes - 1))
                        target_tensor.scatter_(1, targets_int.unsqueeze(1), 1 - label_smoothing)
                else:  # regression or multi_label
                    target_tensor = torch.tensor(targets, dtype=torch.float32)

                # Apply data collation if requested
                if use_data_collator == "True":
                    collator = DataCollatorWithPadding(
                        tokenizer=None,  # Not needed for target tensors
                        padding=padding,
                        max_length=None,
                        pad_to_multiple_of=pad_to_multiple_of if pad_to_multiple_of > 0 else None,
                        return_tensors=return_tensors
                    )
                    
                    # Prepare inputs for collation
                    collated_outputs = collator([{"labels": t} for t in target_tensor])
                    target_tensor = collated_outputs["labels"]
                else:
                    collated_outputs = {"labels": target_tensor}

                # Apply tensor processing options
                def prepare_tensor(tensor):
                    if detach_tensor == "True":
                        tensor = tensor.detach()
                    if make_clone == "True":
                        tensor = tensor.clone()
                    if requires_grad == "True":
                        tensor = tensor.requires_grad_(True)
                    return tensor

                target_tensor = prepare_tensor(target_tensor)

                # Calculate label statistics
                label_info = {
                    "num_unique": len(np.unique(targets)),
                    "value_counts": dict(zip(*np.unique(targets, return_counts=True))),
                    "min": float(target_tensor.min()),
                    "max": float(target_tensor.max()),
                    "mean": float(target_tensor.float().mean())
                }

                # Create detailed info message
                info = (
                    f"Processed {len(targets)} targets\n"
                    f"Type: {target_type}\n"
                    f"Encoding: {encoding}\n"
                    f"Output shape: {list(target_tensor.shape)}\n"
                    f"Label mapping: {'Custom' if custom_label_map != '{}' else 'Auto-generated' if create_label_maps == 'True' else 'None'}\n"
                    f"Data collation: {use_data_collator}\n"
                    f"Number of classes: {num_classes}\n"
                    f"Target value range: [{float(target_tensor.min()):.2f}, {float(target_tensor.max()):.2f}]\n"
                    f"Unique values: {label_info['num_unique']}\n"
                    f"Tensor properties:\n"
                    f"- Detached: {detach_tensor}\n"
                    f"- Requires grad: {requires_grad}\n"
                    f"- Cloned: {make_clone}\n"
                    f"- Dtype: {target_tensor.dtype}"
                )

                return (target_tensor, info, label_info, collated_outputs)

        except Exception as e:
            import traceback
            error_msg = f"Error processing targets: {str(e)}\n{traceback.format_exc()}"
            return (torch.empty(0), error_msg, {}, {})            

           


class NntDatasetToTensor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": ("DATASET",),
                "column_name": ("STRING", {
                    "default": "label",
                    "multiline": False,
                    "placeholder": "Column name to convert"
                })
            }
        }

    RETURN_TYPES = ("TENSOR", "STRING")
    FUNCTION = "convert_to_tensor"
    CATEGORY = "NNT Neural Network Toolkit/Data Processing"

    def convert_to_tensor(self, dataset, column_name):
        try:
            import torch
            with torch.inference_mode(False), torch.set_grad_enabled(True):
                # Get data from specified column
                data = dataset[column_name]
                
                # Convert to tensor
                tensor = torch.tensor(data)
                
                # Create info message
                info = (f"Converted column '{column_name}' to tensor\nShape: {list(tensor.shape)}\nDtype: {tensor.dtype}\n"
                       f"First row dataset: {dataset[column_name][0]}"
                        )
                return (tensor, info)
            
        except Exception as e:
            error_msg = f"Error converting to tensor: {str(e)}"
            return (torch.empty(0), error_msg)

