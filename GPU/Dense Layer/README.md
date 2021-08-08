# Dense_Layer - GPU

It was created with reference to the Dense layer used in Keras. If you are using Dense layer in your model, you have to use the layer here.
For example, let's assume that we take a vector of 1x100 from the input. Also, we have a model of 3 Dense layers and the first layer is 100, the second layer is 20, and the output layer is 1 neuron.

First, let's define the input vector. If bias is used in dense layers, 1.0 bias value should be added to the input vector. Keras use bias value by default in Dense layers -> [Source](https://keras.io/api/layers/core_layers/dense/). The bias value is 1.0 by default.

```ini
CpuGpuMat inputImage(1, 100);         // 1x100 inputs + 1 bias => 101 inputs
```

Then, Dense layer, which is input vector size and consists of 100 neurons, is created.

``` ini
Dense dense(100, 1, 100);             // Dense dense(neurons, input.Rows, input.Cols)
Dense dense1(20, dense.Result);       // Each layer gets the result values of the previous layer
Dense dense2(1, dense1.Result, true); // Dense dense2(neurons, prelayer_result, isEndLayer) // Yeaaap, dense2 is end layer
```

Next Dense layers are created by considering the dimensions of the result matrices of the previous layer. Since the output layer cannot have a bias value in the result matrix, the last layer's useBias value is set to false.

Then, "Output" defined for it to able to read the output of the last layer

```ini
CpuGpuMat output(dense2.Result, 1);
```

Then the weight of the dense layers are loaded.

```ini
std::string denseKernel = ".\\dense\\kernel.txt";
std::string denseBias = ".\\dense\\bias.txt";

std::string dense1Kernel = ".\\dense_1\\kernel.txt";
std::string dense1Bias = ".\\dense_1\\bias.txt";

std::string dense2Kernel = ".\\dense_2\\kernel.txt";
std::string dense2Bias = ".\\dense_2\\bias.txt";

// load kernel and bias weights to ram
dense.Load(denseKernel, denseBias);
dense1.Load(dense1Kernel, dense1Bias);
dense2.Load(dense2Kernel, dense2Bias);
```

The then, ram memory is copied into the graphics card memory.

```ini
dense.Host2Device();
dense1.Host2Device();
dense2.Host2Device();
inputImage.Host2Device();
```

Next, ANN is created.

```ini
dense.Apply(&inputImage);
...
dense1.Apply(&dense.Result);
...
dense2.Apply(&dense1.Result);
...
```

The output of the last layer is copied to ram memory so that you can see the result.

```ini
output.Device2Host();
```

Finally the result values is show

```ini
float* variancePredict = (float*)output.CpuP;
std::cout << variancePredict[0];
```
