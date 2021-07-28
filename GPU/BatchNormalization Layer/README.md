# CUDA_Keras_BatchNormalization_Layer

It was created with reference to the Batch Normalization layer used in Keras. If you are using Batch Normalization layer in your model, you have to use the layer here.
As an example, let's say you want to use Batch Normalization layer after a Dense layer. Also, suppose your input layer gets 1x625 vector.

First, let's define the input vector. If bias is used in dense layers, 1.0 bias value should be added to the input vector. For this (by default) useBias = true is used. Keras use bias value by default in Dense layers -> [Source](https://keras.io/api/layers/core_layers/dense/)
```
CpuGpuMat inputImage(1, 625);   // CpuGpuMat inputVector(1, cols, bias=true)
```

Then, Dense layer, which is input vector size and consists of 100 neurons, is created.
```
Dense dense(100, inputImage.Rows, inputImage.Cols);   // Dense dense(neurons, inputVector.Rows, inputVector.Cols, bias=true)
```

Batch Normalization layer are then created. Batch Normalization layer is created by considering the result matrix of the layer to be applied.

```
BatchNormalization batchNorm(dense.Result.Rows, dense.Result.Cols);
```

Batch Normalization layer are dependent on the output size of the dense layer to be applied. Then the weight of the dense and batchnormalization layer are loaded.

```
std::string denseKernel = ".\\dense\\kernel.txt";
std::string denseBias = ".\\dense\\bias.txt";

dense.load(denseKernel, denseBias);
```

```
std::string batchNormBeta = ".\\batch_normalization\\beta.txt";
std::string batchNormGamma = ".\\batch_normalization\\gamma.txt";
std::string batchNormMovingMean = ".\\batch_normalization\\moving_mean.txt";
std::string batchNormMovingVariance = ".\\batch_normalization\\moving_variance.txt";

batchNorm.load(batchNormBeta, batchNormGamma, batchNormMovingMean, batchNormMovingVariance);
```

The then, ram memory is copied into the graphics card memory.

```
dense.host2Device();
batchNorm.host2Device();
inputImage.host2Device();
```

Next, ANN is created.

```
dense.apply(&inputImage);
batchNorm.apply(&dense.Result);
```

The output of the last layer is copied to ram memory so that you can see the result.
```
dense.Result.device2Host();
```

Finally the result values is show
```
float* resultP = (float*)dense.Result.CpuP;
for (int i = 0; i < inputImage.Size; i++)			// last value is bias = 1.0 value
  std::cout<< resultP[i] << std::endl;
```

You can also take a look at the example in **main.cpp**.
