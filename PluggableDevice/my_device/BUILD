cc_binary(
    name = "libdemo_plugin.so",
    linkshared = True,
    linkstatic = False, # Ensures we link dynamically against external libs
    visibility = ["//visibility:public"],
    deps = [
        "//my_device/src:plugin_device",
        "//my_device/src:plugin_kernel",
    ],
    linkopts = [
        "-L/opt/conda/lib/python3.10/site-packages/tensorflow/python",
        "-l:_pywrap_tensorflow_internal.so",
        "-Wl,-rpath,/opt/conda/lib/python3.10/site-packages/tensorflow/python",
    ],
)

config_setting(
    name = "linux_x86_64",
    values = {"cpu": "k8"},
    visibility = ["//visibility:public"],
)

cc_library(
    name = "core",
    visibility = ["//visibility:public"],
    deps = [
        "//my_device/src/device/tpu_sim:tpu_sim_device_impl",
        "@local_config_tf//:tf_header_lib",
    ],
    alwayslink = True,
)
