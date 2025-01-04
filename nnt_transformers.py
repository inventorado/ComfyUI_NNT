import torch
import torch.nn as nn
import math
import os

# Transformer configuration constants
TRANSFORMER_ACTIVATIONS = [
    "relu",
    "gelu",
    "silu",
    "tanh"
]

TRANSFORMER_ENCODING_TYPES = [
    "sinusoidal",
    "learned",
    "rotary",
    "alibi"
]

ATTENTION_TYPES = [
    "dot_product",
    "additive",
    "scaled_dot_product",
    "relative",
    "local"
]

# Model dimension ranges
MODEL_DIM_CONFIG = {
    "d_model": {
        "default": 512,
        "min": 64,
        "max": 2048,
        "step": 64
    },
    "embed_dim": {
        "default": 512,
        "min": 64,
        "max": 2048,
        "step": 64
    },
    "dim_feedforward": {
        "default": 2048,
        "min": 128,
        "max": 8192,
        "step": 128
    },
    "max_seq_length": {
        "default": 512,
        "min": 16,
        "max": 2048,
        "step": 16
    }
}

# Attention configuration ranges
ATTENTION_CONFIG = {
    "num_heads": {
        "default": 8,
        "min": 1,
        "max": 32,
        "step": 1
    },
    "dropout": {
        "default": 0.1,
        "min": 0.0,
        "max": 0.9,
        "step": 0.1
    },
    "window_size": {
        "default": 128,
        "min": 16,
        "max": 512,
        "step": 16
    }
}

class NntDefineTransformerEncoderLayer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "d_model": ("INT", MODEL_DIM_CONFIG["d_model"]),
                "nhead": ("INT", ATTENTION_CONFIG["num_heads"]),
                "dim_feedforward": ("INT", MODEL_DIM_CONFIG["dim_feedforward"]),
                "dropout": ("FLOAT", ATTENTION_CONFIG["dropout"]),
                "activation": (TRANSFORMER_ACTIVATIONS, {
                    "default": "relu"
                }),
                "batch_first": (["True", "False"], {
                    "default": "True"
                }),
                "norm_first": (["True", "False"], {
                    "default": "False"
                }),
            },
            "optional": {
                "LAYER_STACK": ("LIST",),
            },
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "define_transformer_layer"
    CATEGORY = "NNT Neural Network Toolkit/Transformers"

    def define_transformer_layer(self, d_model, nhead, dim_feedforward, dropout,
                               activation, batch_first, norm_first, LAYER_STACK=None):
        if LAYER_STACK is None:
            LAYER_STACK = []
        else:
            LAYER_STACK = LAYER_STACK.copy()

        layer = {
            'type': 'TransformerEncoder',
            'd_model': d_model,
            'nhead': nhead,
            'dim_feedforward': dim_feedforward,
            'dropout': dropout,
            'activation': activation,
            'batch_first': batch_first == "True",
            'norm_first': norm_first == "True"
        }

        LAYER_STACK.append(layer)
        return (LAYER_STACK,)

class NntDefineVanillaAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "embed_dim": ("INT", MODEL_DIM_CONFIG["embed_dim"]),
                "attention_type": (ATTENTION_TYPES, {
                    "default": "scaled_dot_product"
                }),
                "dropout": ("FLOAT", ATTENTION_CONFIG["dropout"]),
                "use_bias": (["True", "False"], {
                    "default": "True"
                }),
                "add_zero_attn": (["True", "False"], {
                    "default": "False"
                }),
                "batch_first": (["True", "False"], {
                    "default": "True"
                }),
            },
            "optional": {
                "LAYER_STACK": ("LIST",),
            },
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "define_vanilla_attention"
    CATEGORY = "NNT Neural Network Toolkit/Transformers"

    def define_vanilla_attention(self, embed_dim, attention_type, dropout, use_bias,
                               add_zero_attn, batch_first, LAYER_STACK=None):
        if LAYER_STACK is None:
            LAYER_STACK = []
        else:
            LAYER_STACK = LAYER_STACK.copy()

        layer = {
            'type': 'VanillaAttention',
            'embed_dim': embed_dim,
            'attention_type': attention_type,
            'dropout': dropout,
            'use_bias': use_bias == "True",
            'add_zero_attn': add_zero_attn == "True",
            'batch_first': batch_first == "True"
        }

        LAYER_STACK.append(layer)
        return (LAYER_STACK,)

class NntDefineLinearAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "embed_dim": ("INT", MODEL_DIM_CONFIG["embed_dim"]),
                "num_heads": ("INT", ATTENTION_CONFIG["num_heads"]),
                "feature_map": (["elu", "relu", "softmax"], {
                    "default": "elu"
                }),
                "eps": ("FLOAT", {
                    "default": 1e-6,
                    "min": 1e-12,
                    "max": 1e-3,
                    "step": 1e-6
                }),
                "causal": (["True", "False"], {
                    "default": "False"
                }),
                "dropout": ("FLOAT", ATTENTION_CONFIG["dropout"]),
                "batch_first": (["True", "False"], {
                    "default": "True"
                }),
            },
            "optional": {
                "LAYER_STACK": ("LIST",),
            },
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "define_linear_attention"
    CATEGORY = "NNT Neural Network Toolkit/Transformers"

    def define_linear_attention(self, embed_dim, num_heads, feature_map, eps,
                              causal, dropout, batch_first, LAYER_STACK=None):
        if LAYER_STACK is None:
            LAYER_STACK = []
        else:
            LAYER_STACK = LAYER_STACK.copy()

        layer = {
            'type': 'LinearAttention',
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'feature_map': feature_map,
            'eps': eps,
            'causal': causal == "True",
            'dropout': dropout,
            'batch_first': batch_first == "True"
        }

        LAYER_STACK.append(layer)
        return (LAYER_STACK,)

class NntDefineTransformerXLAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "d_model": ("INT", MODEL_DIM_CONFIG["d_model"]),
                "num_heads": ("INT", ATTENTION_CONFIG["num_heads"]),
                "mem_len": ("INT", {
                    "default": 512,
                    "min": 0,
                    "max": 2048,
                    "step": 16
                }),
                "same_length": (["True", "False"], {
                    "default": "False"
                }),
                "clamp_len": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2048,
                    "step": 1
                }),
                "dropout": ("FLOAT", ATTENTION_CONFIG["dropout"]),
                "batch_first": (["True", "False"], {
                    "default": "True"
                }),
            },
            "optional": {
                "LAYER_STACK": ("LIST",),
            },
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "define_transformer_xl_attention"
    CATEGORY = "NNT Neural Network Toolkit/Transformers"

    def define_transformer_xl_attention(self, d_model, num_heads, mem_len, same_length,
                                     clamp_len, dropout, batch_first, LAYER_STACK=None):
        if LAYER_STACK is None:
            LAYER_STACK = []
        else:
            LAYER_STACK = LAYER_STACK.copy()

        layer = {
            'type': 'TransformerXLAttention',
            'd_model': d_model,
            'num_heads': num_heads,
            'mem_len': mem_len,
            'same_length': same_length == "True",
            'clamp_len': clamp_len,
            'dropout': dropout,
            'batch_first': batch_first == "True"
        }

        LAYER_STACK.append(layer)
        return (LAYER_STACK,)

class NntDefineReformerAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "embed_dim": ("INT", MODEL_DIM_CONFIG["embed_dim"]),
                "num_heads": ("INT", ATTENTION_CONFIG["num_heads"]),
                "num_buckets": ("INT", {
                    "default": 32,
                    "min": 8,
                    "max": 128,
                    "step": 8
                }),
                "bucket_size": ("INT", {
                    "default": 64,
                    "min": 16,
                    "max": 256,
                    "step": 16
                }),
                "num_hashes": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 16,
                    "step": 1
                }),
                "causal": (["True", "False"], {
                    "default": "False"
                }),
                "dropout": ("FLOAT", ATTENTION_CONFIG["dropout"]),
                "batch_first": (["True", "False"], {
                    "default": "True"
                }),
            },
            "optional": {
                "LAYER_STACK": ("LIST",),
            },
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "define_reformer_attention"
    CATEGORY = "NNT Neural Network Toolkit/Transformers"

    def define_reformer_attention(self, embed_dim, num_heads, num_buckets, bucket_size,
                                num_hashes, causal, dropout, batch_first, LAYER_STACK=None):
        if LAYER_STACK is None:
            LAYER_STACK = []
        else:
            LAYER_STACK = LAYER_STACK.copy()

        layer = {
            'type': 'ReformerAttention',
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'num_buckets': num_buckets,
            'bucket_size': bucket_size,
            'num_hashes': num_hashes,
            'causal': causal == "True",
            'dropout': dropout,
            'batch_first': batch_first == "True"
        }

        LAYER_STACK.append(layer)
        return (LAYER_STACK,)

class NntDefineLocalAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "embed_dim": ("INT", MODEL_DIM_CONFIG["embed_dim"]),
                "num_heads": ("INT", ATTENTION_CONFIG["num_heads"]),
                "window_size": ("INT", ATTENTION_CONFIG["window_size"]),
                "look_behind": ("INT", {
                    "default": 64,
                    "min": 0,
                    "max": 256,
                    "step": 16
                }),
                "look_ahead": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 256,
                    "step": 16
                }),
                "dropout": ("FLOAT", ATTENTION_CONFIG["dropout"]),
                "autopad": (["True", "False"], {
                    "default": "True"
                }),
                "batch_first": (["True", "False"], {
                    "default": "True"
                }),
            },
            "optional": {
                "LAYER_STACK": ("LIST",),
            },
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "define_local_attention"
    CATEGORY = "NNT Neural Network Toolkit/Transformers"

    def define_local_attention(self, embed_dim, num_heads, window_size, look_behind,
                             look_ahead, dropout, autopad, batch_first, LAYER_STACK=None):
        if LAYER_STACK is None:
            LAYER_STACK = []
        else:
            LAYER_STACK = LAYER_STACK.copy()

        layer = {
            'type': 'LocalAttention',
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'window_size': window_size,
            'look_behind': look_behind,
            'look_ahead': look_ahead,
            'dropout': dropout,
            'autopad': autopad == "True",
            'batch_first': batch_first == "True"
        }

        LAYER_STACK.append(layer)
        return (LAYER_STACK,)

class NntDefinePositionalEncoding:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "d_model": ("INT", MODEL_DIM_CONFIG["d_model"]),
                "max_seq_length": ("INT", MODEL_DIM_CONFIG["max_seq_length"]),
                "dropout": ("FLOAT", ATTENTION_CONFIG["dropout"]),
                "encoding_type": (TRANSFORMER_ENCODING_TYPES, {
                    "default": "sinusoidal"
                }),
                "learnable": (["True", "False"], {
                    "default": "False"
                }),
                "normalize": (["True", "False"], {
                    "default": "True"
                }),
            },
            "optional": {
                "LAYER_STACK": ("LIST",),
            },
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "define_positional_encoding"
    CATEGORY = "NNT Neural Network Toolkit/Transformers"

    def define_positional_encoding(self, d_model, max_seq_length, dropout,
                                 encoding_type, learnable, normalize, LAYER_STACK=None):
        if LAYER_STACK is None:
            LAYER_STACK = []
        else:
            LAYER_STACK = LAYER_STACK.copy()

        layer = {
            'type': 'PositionalEncoding',
            'd_model': d_model,
            'max_seq_length': max_seq_length,
            'dropout': dropout,
            'encoding_type': encoding_type,
            'learnable': learnable == "True",
            'normalize': normalize == "True"
        }

        LAYER_STACK.append(layer)
        return (LAYER_STACK,)

class NntDefineMultiheadAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "embed_dim": ("INT", MODEL_DIM_CONFIG["embed_dim"]),
                "num_heads": ("INT", ATTENTION_CONFIG["num_heads"]),
                "dropout": ("FLOAT", ATTENTION_CONFIG["dropout"]),
                "bias": (["True", "False"], {
                    "default": "True"
                }),
                "add_bias_kv": (["True", "False"], {
                    "default": "False"
                }),
                "add_zero_attn": (["True", "False"], {
                    "default": "False"
                }),
                "batch_first": (["True", "False"], {
                    "default": "True"
                }),
                "kdim": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2048,
                    "step": 64
                }),
                "vdim": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2048,
                    "step": 64
                }),
            },
            "optional": {
                "LAYER_STACK": ("LIST",),
            },
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "define_attention"
    CATEGORY = "NNT Neural Network Toolkit/Transformers"

    def define_attention(self, embed_dim, num_heads, dropout, bias, add_bias_kv,
                        add_zero_attn, batch_first, kdim, vdim, LAYER_STACK=None):
        if LAYER_STACK is None:
            LAYER_STACK = []
        else:
            LAYER_STACK = LAYER_STACK.copy()

        layer = {
            'type': 'MultiheadAttention',
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'dropout': dropout,
            'bias': bias == "True",
            'add_bias_kv': add_bias_kv == "True",
            'add_zero_attn': add_zero_attn == "True",
            'batch_first': batch_first == "True",
            'kdim': kdim if kdim > 0 else None,
            'vdim': vdim if vdim > 0 else None
        }

        LAYER_STACK.append(layer)
        return (LAYER_STACK,)

class NntDefineRelativePositionBias:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_buckets": ("INT", {
                    "default": 32,
                    "min": 8,
                    "max": 128,
                    "step": 8
                }),
                "max_distance": ("INT", {
                    "default": 128,
                    "min": 16,
                    "max": 512,
                    "step": 16
                }),
                "num_heads": ("INT", ATTENTION_CONFIG["num_heads"]),
                "causal": (["True", "False"], {
                    "default": "False"
                }),
            },
            "optional": {
                "LAYER_STACK": ("LIST",),
            },
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "define_relative_position_bias"
    CATEGORY = "NNT Neural Network Toolkit/Transformers"

    def define_relative_position_bias(self, num_buckets, max_distance, num_heads, 
                                    causal, LAYER_STACK=None):
        if LAYER_STACK is None:
            LAYER_STACK = []
        else:
            LAYER_STACK = LAYER_STACK.copy()

        layer = {
            'type': 'RelativePositionBias',
            'num_buckets': num_buckets,
            'max_distance': max_distance,
            'num_heads': num_heads,
            'causal': causal == "True"
        }

        LAYER_STACK.append(layer)
        return (LAYER_STACK,)

class NntDefineAlibiPositionalBias:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_heads": ("INT", ATTENTION_CONFIG["num_heads"]),
                "max_seq_length": ("INT", MODEL_DIM_CONFIG["max_seq_length"]),
                "causal": (["True", "False"], {
                    "default": "False"
                }),
                "slope_multiplier": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "LAYER_STACK": ("LIST",),
            },
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "define_alibi_bias"
    CATEGORY = "NNT Neural Network Toolkit/Transformers"

    def define_alibi_bias(self, num_heads, max_seq_length, causal, 
                         slope_multiplier, LAYER_STACK=None):
        if LAYER_STACK is None:
            LAYER_STACK = []
        else:
            LAYER_STACK = LAYER_STACK.copy()

        layer = {
            'type': 'AlibiPositionalBias',
            'num_heads': num_heads,
            'max_seq_length': max_seq_length,
            'causal': causal == "True",
            'slope_multiplier': slope_multiplier
        }

        LAYER_STACK.append(layer)
        return (LAYER_STACK,)

class NntDefineRotaryPositionalEmbedding:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dim": ("INT", MODEL_DIM_CONFIG["d_model"]),
                "max_freq": ("FLOAT", {
                    "default": 10.0,
                    "min": 1.0,
                    "max": 100.0,
                    "step": 1.0
                }),
                "base": ("FLOAT", {
                    "default": 10000.0,
                    "min": 100.0,
                    "max": 100000.0,
                    "step": 100.0
                }),
                "interpolation_factor": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "LAYER_STACK": ("LIST",),
            },
        }

    RETURN_TYPES = ("LIST",)
    FUNCTION = "define_rotary_embedding"
    CATEGORY = "NNT Neural Network Toolkit/Transformers"

    def define_rotary_embedding(self, dim, max_freq, base, 
                              interpolation_factor, LAYER_STACK=None):
        if LAYER_STACK is None:
            LAYER_STACK = []
        else:
            LAYER_STACK = LAYER_STACK.copy()

        layer = {
            'type': 'RotaryPositionalEmbedding',
            'dim': dim,
            'max_freq': max_freq,
            'base': base,
            'interpolation_factor': interpolation_factor
        }

        LAYER_STACK.append(layer)
        return (LAYER_STACK,)

class NntTextBatchProcessor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "texts": ("STRING", {
                    "multiline": True,
                    "default": "Text 1\n---\nText 2\n---\nText 3"
                }),
                "separator": ("STRING", {
                    "default": "---",
                    "multiline": False
                }),
                "max_length": ("INT", {
                    "default": 256,
                    "min": 16,
                    "max": 2048,
                    "step": 1
                }),
                "batch_size": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 512,
                    "step": 1
                }),
                "tokenizer": (["bert-base-uncased", "distilbert-base-uncased"], {
                    "default": "bert-base-uncased"
                }),
                "output_dtype": (["float32", "float64", "long", "int32"], {
                    "default": "long"
                })
            }
        }

    RETURN_TYPES = ("TENSOR", "INT", "STRING")
    RETURN_NAMES = ("token_batches", "num_batches", "info")
    FUNCTION = "process_batch"
    CATEGORY = "NNT Neural Network Toolkit/Text"

    def _convert_dtype(self, tensor, dtype_str):
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "long": torch.long,
            "int32": torch.int32
        }
        return tensor.to(dtype_map[dtype_str])

    def process_batch(self, texts, separator, max_length, batch_size, tokenizer, output_dtype):
        try:
            from transformers import AutoTokenizer
            import torch

            # Split texts
            text_list = [t.strip() for t in texts.split(separator) if t.strip()]
            
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

            # Process in batches
            all_batches = []
            for i in range(0, len(text_list), batch_size):
                batch_texts = text_list[i:i + batch_size]
                inputs = tokenizer(
                    batch_texts,
                    max_length=max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                # Convert to requested dtype
                all_batches.append(self._convert_dtype(inputs['input_ids'], output_dtype))

            # Stack all batches
            if all_batches:
                batched_tokens = torch.cat(all_batches, dim=0)
            else:
                batched_tokens = torch.zeros(1, max_length, dtype=self._convert_dtype(torch.tensor([0]), output_dtype).dtype)

            info = f"Processed {len(text_list)} texts in {len(all_batches)} batches\n"
            info += f"Token shape: {batched_tokens.shape}\n"
            info += f"Output dtype: {output_dtype}\n"
            info += f"Tokenizer: {tokenizer.name_or_path}"

            return (batched_tokens, len(all_batches), info)

        except Exception as e:
            import traceback
            error_msg = f"Error processing batch: {str(e)}\n{traceback.format_exc()}"
            return (torch.zeros(1, max_length, dtype=self._convert_dtype(torch.tensor([0]), output_dtype).dtype), 0, error_msg)