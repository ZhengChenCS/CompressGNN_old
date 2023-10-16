# CompressGNN
This is the anonymous repository for submitting paper **CompressGNN: Accelerating Graph Neural Network Training via
Hierarchical Compression**.

## Code Structure

The project's code base is organized in the following directory structure.

```shell
.
├── benchmark
├── dataset
├── install.sh
├── layer
├── LICENSE
├── loader
├── model
├── README.md
├── requirements.txt
└── src
```

## Installaction

To use this repository, please follow the steps below to install the required dependencies.

### Prerequisites

- Python (version 3.8.13)
- pip (version 23.0.1)
- cuda (version 11.6)

### Installing Dependencies

1. Clone the repository

2. Navigate to the project directory

3. Install the required dependencies using pip:

```shell
pip install -r requirements.txt
```
This command will install all the necessary libraries and packages, including:

- numpy
- pytorch
- Pytorch Geometric (PyG)
- Deep Graph Library (DGL)
- torch_scatter
- torch_sparse
- pybind
- ...

4. Install CompressGNN

```shell
bash install.sh
```

## Run Evaluation


### End-to-end performance

```shell
cd benchmark/end2end
bash run.sh
```

### Propagate performance

- Speedup

```shell
cd benchmark/propagate/propagate
bash speedup.sh
```

- Performance with different feature dimension

```shell
cd benchmark/propagate/propagate
bash feature_scale.sh
```

- Peak memory

```shell
cd benchmark/propagate/peak_memory
bash run.sh
```

### Transformation performance

- Time and accuracy

```shell
cd benchmark/transform/time_accu
bash run.sh
```

- Time breakdown

```shell
cd benchmark/transform/time_breakdown
bash run.sh
```







