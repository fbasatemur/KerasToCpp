# BatchNormalization_Layer - GPU

It was created with reference to the Batch Normalization layer used in Keras. If you are using Batch Normalization layer in your model, you have to use the layer here.
As an example, let's say you want to use Batch Normalization layer after a Dense layer. Also, suppose your input layer gets 1x100 vector.

First, let's define the input vector.

```ini
CpuGpuMat inputImage(1, 100);      // 1x100 inputs + 1 bias => 101 inputs
```

Then, Batch Normalization layer is created. The Batch Normalization layer is created depending on the properties of the matrix to be applied.

```ini
BatchNormalization batchNorm(inputImage.Rows, inputImage.Cols);
```

Then the weight of the Batch Normalization layer is loaded.

```ini
std::string batchNormBeta = ".\\batch_normalization\\beta.txt";
std::string batchNormGamma = ".\\batch_normalization\\gamma.txt";
std::string batchNormMovingMean = ".\\batch_normalization\\moving_mean.txt";
std::string batchNormMovingVariance = ".\\batch_normalization\\moving_variance.txt";

batchNorm.Load(batchNormBeta, batchNormGamma, batchNormMovingMean, batchNormMovingVariance);
```

The then, ram memory is copied into the graphics card memory.

```ini
batchNorm.Host2Device();
inputImage.Host2Device();
```

Next, Batch Normalization apply.

```ini
batchNorm.Apply(&inputImage);
```

The matrix given as input is copied to ram memory to read the results

```ini
inputImage.Device2Host();
```

Finally the result values is show

```ini
float* resultP = (float*)inputImage.CpuP;
for (int i = 0; i < inputImage.Size; i++)			// last value is bias = 1.0 value
  std::cout<< resultP[i] << std::endl;
```

You can also take a look at the example in **main.cpp**.
