# Neural Network Toolkit (NNT) for ComfyUI

The **Neural Network Toolkit (NNT)** is an extensive set of custom ComfyUI nodes for designing, training, and fine-tuning neural networks. This toolkit allows defining models, layers, training workflows, transformers, and tensor operations in a visual manner using nodes.

This toolkit is especially useful in educational environments for learning and explaining neural network principles and architecture without code. Even people with no programming experience can use it to experiment with neural networks. I created this project as a practical learning tool while studying neural networks.

NNT allows you to quickly create and train various models, experiment with their structure, and observe the effect of changing their parameters. While it's a powerful learning and prototyping tool, please note it's a work in progress and not meant to replace PyTorch coding for production environments.

## License

This project is licensed under the GNU General Public License v3.0 - see [LICENSE](LICENSE) for details.

## Key Features

-   Visual node-based neural network design
-   Support for various layer types (Dense, Conv, RNN, Transformer)
-   Interactive training and fine-tuning capabilities
-   Real-time visualization of model architecture and training metrics
-   Tensor manipulation and analysis tools
-   Educational focus with immediate visual feedback
-   No coding required for basic operations

## Installation

### Prerequisites

-   ComfyUI installed and working
-   Python 3.8+
-   CUDA-compatible GPU (recommended)
-   PyTorch 2.0+
-   Basic understanding of neural networks

cd ComfyUI/custom_nodes
git clone https://github.com/inventorado/ComfyUI_NNT.git
cd ComfyUI_NNT
pip install -r requirements.txt

## Node List and Descriptions

### Models

#### **NntCompileModel**

Compiles a neural network model from a layer stack.

-   **Inputs**: Layer stack, activation function, normalization, weight initialization.
-   **Options**: Compile mode, activation parameters.
-   **Outputs**: Compiled model, report, script.

#### **NntTrainModel**

Trains a model using provided training data.

-   **Inputs**: Model, training data, target data, hyperparameters.
-   **Options**: Batch size, epochs, loss function, optimizer.
-   **Outputs**: Trained model, training log, metrics.

#### **NntFineTuneModel**

Fine-tunes an existing model on new datasets.

-   **Inputs**: Model, training/validation data, learning rate, batch size.
-   **Options**: Scheduler settings, early stopping.
-   **Outputs**: Fine-tuned model, training summary.

#### **NntSaveModel**

Saves the model in various formats.

-   **Inputs**: Model, filename, directory.
-   **Options**: Save format, quantization type, optimizer inclusion.
-   **Outputs**: Model, save report.

#### **NntLoadModel**

Loads a model from a specified file.

-   **Inputs**: File path, load format.
-   **Options**: Device selection.
-   **Outputs**: Loaded model.

#### **NntAnalyzeModel**

Analyzes model layers, memory usage, and complexity.

-   **Inputs**: Model, input shape, batch size.
-   **Outputs**: Analysis report.

#### **NntVisualizeGraph**

Visualizes the computation graph of the model.

-   **Inputs**: Model.
-   **Outputs**: Graph visualization image.

#### **NntEditModelLayers**

Edits layers in an existing model.

-   **Inputs**: Model, layer editing instructions.
-   **Options**: Freeze/unfreeze layers, apply quantization, or modify weights.
-   **Outputs**: Edited model, operation summary.

#### **NntMergeExtendModel**

Merges or extends models with new layers.

-   **Inputs**: Two models or one model with a layer stack.
-   **Options**: Merge strategies, additional layer configurations.
-   **Outputs**: Merged/extended model, merge summary.
* * *

### Layers

#### **NntInputLayer**

Defines the input layer for the model.

-   **Inputs**: Input shape as a list.
-   **Outputs**: Layer stack.

#### **NntDefineDenseLayer**

Defines a fully connected (dense) layer.

-   **Inputs**: Number of nodes, activation function, weight initialization.
-   **Options**: Bias settings, normalization, dropout rate.
-   **Outputs**: Layer stack, number of nodes.

#### **NntDefineConvLayer**

Defines convolutional layers with extensive configurations.

-   **Inputs**: Kernel size, stride, padding, dilation, channels.
-   **Options**: Weight initialization, normalization, dropout rate.
-   **Outputs**: Layer stack.

#### **NntDefinePoolingLayer**

Configures pooling layers (max, average, or adaptive).

-   **Inputs**: Pooling type, kernel size, stride.
-   **Options**: Padding, advanced settings for fractional or LPPool.
-   **Outputs**: Layer stack.

#### **NntDefineRNNLayer**

Defines recurrent layers like RNNs, LSTMs, and GRUs.

-   **Inputs**: Input size, hidden size, number of layers.
-   **Options**: Nonlinearity (tanh/relu), dropout, bidirectional settings.
-   **Outputs**: Layer stack.

#### **NntDefineFlattenLayer**

Flattens input tensors for fully connected layers.

-   **Inputs**: None.
-   **Outputs**: Layer stack.

#### **NntDefineNormLayer**

Adds normalization layers (batch, layer, instance).

-   **Inputs**: Type of normalization, number of features.
-   **Options**: Momentum, epsilon, affine parameters.
-   **Outputs**: Layer stack.

#### **NntDefineActivationLayer**

Configures activation functions for layers.

-   **Inputs**: Activation type (ReLU, Sigmoid, Tanh, etc.).
-   **Options**: In-place operations, additional parameters for specific activations.
-   **Outputs**: Layer stack.
* * *

### Transformers

#### **NntDefineTransformerEncoderLayer**

Defines a transformer encoder layer.

-   **Inputs**: Model dimensions, number of heads, feedforward size.
-   **Options**: Batch-first processing, dropout, activation function.
-   **Outputs**: Layer stack.

#### **NntDefineMultiheadAttention**

Defines multi-head attention layers.

-   **Inputs**: Embedding dimension, number of heads, dropout.
-   **Options**: Add bias, batch-first processing.
-   **Outputs**: Layer stack.

#### **NntDefineVanillaAttention**

Implements basic attention mechanisms.

-   **Inputs**: Embedding dimension, attention type, dropout.
-   **Options**: Batch processing, zero attention.
-   **Outputs**: Layer stack.

#### **NntDefineLinearAttention**

Implements linear attention for efficient computation.

-   **Inputs**: Embedding dimension, feature map, dropout.
-   **Options**: Causality, epsilon for numerical stability.
-   **Outputs**: Layer stack.

#### **NntDefineReformerAttention**

Adds Reformer-style attention.

-   **Inputs**: Bucketing configuration, dropout, number of heads.
-   **Options**: Causality, hashing parameters.
-   **Outputs**: Layer stack.

#### **NntDefinePositionalEncoding**

Defines positional encodings for sequence data.

-   **Inputs**: Encoding type, sequence length, dropout.
-   **Options**: Normalize, learnable parameters.
-   **Outputs**: Layer stack.
* * *

### Tensors

#### **NntTensorToText**

Converts tensors to human-readable text.

-   **Inputs**: Tensor, format options, precision.
-   **Options**: Max elements for output.
-   **Outputs**: Text representation.

#### **NntTextToTensor**

Parses text into a PyTorch tensor.

-   **Inputs**: Text data, data type, device.
-   **Options**: Gradient requirements.
-   **Outputs**: Tensor.

#### **NntRandomTensorGenerator**

Generates random tensors based on various distributions.

-   **Inputs**: Distribution type, shape, data type.
-   **Options**: Seed, min/max range for values.
-   **Outputs**: Tensor, generation summary.

#### **NntTensorElementToImage**

Converts tensor elements to images.

-   **Inputs**: Tensor, image dimensions, channels.
-   **Options**: Clamp range, reshape options.
-   **Outputs**: Image tensor.

#### **NntDataLoader**

Loads datasets in various formats (text, images, paired data).

-   **Inputs**: Data source, file path, data type.
-   **Options**: Normalization, batch-first processing.
-   **Outputs**: Tensors, paired data (optional), info message.
* * *

### Visualization

#### **NntVisualizeTrainingMetrics**

Plots training metrics over time (loss, accuracy).

-   **Inputs**: Metrics data.
-   **Options**: Plot dimensions, smoothing factors.
-   **Outputs**: Visualization image.

#### **NntSHAPSummaryNode**

Generates SHAP explanations for model predictions.

-   **Inputs**: Model, sample data, plot type.
-   **Options**: Background sample size.
-   **Outputs**: SHAP summary report, visualization plot.

#### **NntVisualizePredictions**

Visualizes predictions with confusion matrices and error analysis.

-   **Inputs**: Model, input data, target data.
-   **Options**: Classification or regression task type.
-   **Outputs**: Report, main plot, loss/error plot, metrics.
* * *

## Getting Started

### Step 1: Build Layers

Use layer nodes (e.g., `NntDefineDenseLayer`) to define the architecture of your network.

### Step 2: Compile Model

Pass the layer stack to `NntCompileModel` to create a trainable model.

### Step 3: Train Model

Use `NntTrainModel` with training and target data to train your model.

### Step 4: Visualize Results

Analyze performance using visualization nodes (e.g., `NntVisualizeTrainingMetrics`).

* * *

## Contributing

We welcome contributions! Submit issues, feature requests, or pull requests to enhance the toolkit.