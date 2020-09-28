# CUDA_Keras_Dense_Layer

It was created with reference to the Dense layer used in Keras. If you are using Dense layer in your model, you have to use the layer here.
For example, let's assume that we take a vector of 1x625 from the input. Also, we have a model of 3 Dense layers and the first layer is 100, the second layer is 20, and the output layer is 1 neuron.

First, let's define the input vector. If bias is used in dense layers, 1.0 bias value should be added to the input vector. For this (by default) useBias = true is used. Keras use bias value by default in Dense layers -> [Source](https://keras.io/api/layers/core_layers/dense/)
```
CpuGpuMat inputImage(1, 625);   // CpuGpuMat inputVector(1, cols, bias=true)
```

Then, Dense layer, which is input vector size and consists of 100 neurons, is created.

```
Dense dense(100, inputImage.Rows, inputImage.Cols);         // Dense dense(neurons, inputVector.Rows, inputVector.Cols, bias=true)
Dense dense1(20, dense.Result.Rows, dense.Result.Cols);     
Dense dense2(1, dense1.Result.Rows, dense1.Result.Cols, false);
```

Next Dense layers are created by considering the dimensions of the result matrices of the previous layer. Since the output layer cannot have a bias value in the result matrix, the last layer's useBias value is set to false.


Then the weight of the dense layers are loaded.

```
std::string denseKernel = ".\\dense\\kernel.txt";
std::string denseBias = ".\\dense\\bias.txt";

std::string dense1Kernel = ".\\dense_1\\kernel.txt";
std::string dense1Bias = ".\\dense_1\\bias.txt";

std::string dense2Kernel = ".\\dense_2\\kernel.txt";
std::string dense2Bias = ".\\dense_2\\bias.txt";

// load kernel and bias weights to ram
dense.load(denseKernel, denseBias);
dense1.load(dense1Kernel, dense1Bias);
dense2.load(dense2Kernel, dense2Bias);
```

The then, ram memory is copied into the graphics card memory.

```
dense.host2Device();
dense1.host2Device();
dense2.host2Device();
inputImage.host2Device();
```

Next, ANN is created.

```
dense.apply(&inputImage);
...
dense1.apply(&dense.Result);
...
dense2.apply(&dense1.Result);
...
```

The output of the last layer is copied to ram memory so that you can see the result.
```
dense2.Result.device2Host();
```

Finally the result values is show
```
float* variancePredict = (float*)dense2.Result.CpuP;
std::cout << variancePredict[0];
```
