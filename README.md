# deep-manifold-embedding

> This repository is an extension of Caffe for the paper "Deep Manifold Embedding for Hyperspectral Image Classification" (IEEE TCYB). [[paper](https://arxiv.org/abs/1912.11264)]

## Manifold Learning

> The Code for implementation of manifold learning is provided by Josh Tenenbaum [[code](https://github.com/deblearn/Isomap)]. 

## Installation

1. Install prerequisites for `Caffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

2. Add the header and source file in the Caffe project.

3. Modify caffe.proto

   ```c
   message IsomapManifoldLearningLossParameter {
     optional float margin = 1 [default = 1.0];
     optional int32 num_output = 2 [default = 5];
     optional int32 class_k = 5 [default = 1];
     optional float inter_weight = 3 [default = 10e-4];
     optional float intra_weight = 4 [default = 10e-5];
     optional int32 axis = 6 [default = 1];
     //optional FillerParameter center_filler = 7; // The filler for the centers
   }
   message IsomapSoftmaxWithLossParameter {
     optional int32 class_k = 1 [default = 1]; // The number of outputs for the layer
   }
   ```

4. Compile Caffe framework.

## Example

> `Example\` shows the examples of`solver.prototxt` and `train_val.prototxt`.

## Citing this work

If you find this work helpful for your research, please consider citing:

    @article{gong2021,
        Author = {Zhiqiang Gong and Weidong Hu and Xiaoyong Du and Ping Zhong and Panhe Hu},
        Title = {Deep Manifold Embedding for Deep Learning in Hyperspectral Image Classification},
        Booktitle = {IEEE Transactions on Cybernetics},
        Year = {2021}
    }