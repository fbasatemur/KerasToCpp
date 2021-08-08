# Sigmoid_Activation - GPU

It was created with reference to the Sigmoid activation used in Keras. If you are using Sigmoid activation in your model, you have to use the layer here.
For example, let's say you want to apply Sigmoid activation to a vector of 1x100.

First, let's define the input vector.

```ini
CpuGpuMat inputImage(1, 100);   	// inputVector(1, cols, useMemPin = false)
```

We created a vector of 1x100, but we get a 1x101 vector with the bias value. Therefore, value will be assigned 100 times. This does not need to be taken into account when Sigmoid activation is used with other layers.

```ini
// set values to input vector
float* inputP = (float*)inputImage.CpuP;
for (int i = 0; i < inputImage.Size - 1; i++)		  // last value is bias = 1.0 value
	inputP[i] = -1 * (float)i;
```

The then, ram memory is copied into the graphics card memory.

```ini
inputImage.Host2Device();
```

Apply Sigmoid activation

```ini
gpuSigmoid(&inputImage);
```

The Sigmoid activation function is applied on the input vector. Therefore, the gpu memory is copied into the ram memory so that the results can be observed.

```ini
inputImage.Device2Host();
```

Finally the result values is show

```ini
float* resultP = (float*)inputImage.CpuP;
for (int i = 0; i < inputImage.Size; i++)			// last value is bias = 1.0 value
	std::cout << resultP[i] << std::endl;
```
