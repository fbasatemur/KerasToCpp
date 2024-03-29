# KerasToCpp - GPU

 All layer and activation functions are designed to run using GPU. This will provide you with the fastest testing environment. The layers created so far are as follows:

## Supported Keras layers

- ***Layers***
  - :white_check_mark: Dense
  - :white_check_mark: BatchNormalization

- ***Activations***
  - :white_check_mark: Sigmoid
  - :white_check_mark: ReLU

A sample project created using all layers can be accessed here: [Image_Quality_Assessment_with_ANN](https://github.com/fbasatemur/Image_Quality_Assessment_with_ANN)

### Requirements

You will do your testing in parallel using the GPU. Therefore, you will need the following device and driver requirements.

* **CUDA == 10.2**: https://developer.nvidia.com/cuda-toolkit-archive
* **cuDNN == v8.0.2**: https://developer.nvidia.com/rdp/cudnn-archive
* **GPU with CC >= 3.0**: https://en.wikipedia.org/wiki/CUDA#GPUs_supported
* **Windows MSVC 2017/2019**


### How to use Keras model weights in the C environment ?

Keras weights are in hdf5 file format. I assume you got the model record as .json and .h5.
You can create your model training weights as follows:

```ini
# keras library import  for Saving and loading model and weights

from keras.models import model_from_json
from keras.models import load_model

# serialize model to JSON
#  the keras model which is trained is defined as 'model' in this example
model_json = model.to_json()

with open("model_save_json.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model_save_weight.h5")
```

It is converted to a text file for use with the C environment. You can do it as follows:

```ini
python h5_to_txt.py model_save_weight.h5
```

Each layer in the model will be saved in a folder and their weight in it. The text files will then be loaded into the model layers.
