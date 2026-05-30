# TensorFlow PluggableDevice demo
This sample is a simple demo shows how to implement, build, install and run a TensorFlow PluggableDevice as a plugin.

## Supported OS
* Linux

## Setting

- Ubuntu 22.04.5 LTS
- python 3.10.3
- bazel 7.7.0  [download here](https://github.com/bazelbuild/bazel/releases/tag/7.7.0)
  ```
  mv bazel-7.7.0-linux-x86_64 $CONDA_PREFIX/bin/bazel
  chmod +x $CONDA_PREFIX/bin/bazel
  bazel clean --expunge
  ```
- tensorflow 2.20.0
  ```
  $ pip install tensorflow
  ```
- build 1.4.0
  ```
  $ pip install build
  ```

## Build and Run

see all_build_and_test.sh
then `./all_build_and_test.sh`


### Linux

1. In the plug-in `PluggableDeviceSample` code folder, configure the build options:
    ```
    $ ./configure 

    Please specify the location of python. [Default is /opt/conda/bin/python]: 


    Found possible Python library paths:
      /opt/conda/lib/python3.10/site-packages
    Please input the desired Python library path to use.  Default is [/opt/conda/lib/python3.10/site-packages]

    Do you wish to build TensorFlow plug-in with MPI support? [y/N]: 
    No MPI support will be enabled for TensorFlow plug-in.

    Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]: 


    done
    Configuration finished
    ```

3. Built the plug-in with bazel
    ```
    bazel build  -c opt //my_device/tools/pip_package:build_pip_package --verbose_failures
    ```
4. Then generate a python wheel and install it.
    ```
    bazel-bin/my_device/tools/pip_package/build_pip_package .
    pip install my_device-0.0.1-cp310-cp310-linux_x86_64.whl
    ```
5. Now we can run the TensorFlow with plug-in device enabled.
    ```
    $ python
    >>> import tensorflow as tf
    >>> tf.config.list_physical_devices()
    [PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:MY_DEVICE:0', device_type='MY_DEVICE')]
    ```
