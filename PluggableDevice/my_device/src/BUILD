# my_device/src/BUILD

package(
    licenses = ["notice"],
)

cc_library(
    name = "plugin_device",
    srcs = ["plugin_device.cc"],
    hdrs = ["plugin_device.h"],
    linkstatic = True,
    alwayslink = True,
    visibility = ["//visibility:public"],
    deps = [
        "//my_device/src/device/tpu_sim:tpu_sim_device_impl",
        "@local_config_tf//:tf_header_lib",
    ],
)

cc_library(
    name = "plugin_kernel",
    srcs = ["plugin_kernel.cc"],
    hdrs = ["plugin_kernel.h"],
    linkstatic = True,
    alwayslink = True,
    visibility = ["//visibility:public"],
    deps = [
        ":plugin_device",
        "//my_device/src/kernels/tpu_sim:tpu_sim_kernel_impl",
        "@local_config_tf//:tf_header_lib",
    ],
)
