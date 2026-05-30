load("//my_device:demo_plugin.bzl", "tf_copts")

package(
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "relu_op",
    srcs = ["relu_op.cc"],
    copts = tf_copts(),
    linkstatic = True,
    alwayslink = True,
    visibility = ["//visibility:public"],
    deps = [
        "//my_device:core",
    ],
)

TPU_SIM_KERNELS = [
    ":relu_op",
]

cc_library(
    name = "tpu_sim_kernel_impl",
    srcs = ["tpu_sim_kernel_init.cc"],
    hdrs = [
        "tpu_sim_kernel_init.h",
        "//my_device/src/device/tpu_sim:tpu_sim_device_plugin.h",
    ],
    linkstatic = True,
    alwayslink = True,
    visibility = ["//visibility:public"],
    deps = [
        "//my_device:core",
    ] + TPU_SIM_KERNELS,
)
