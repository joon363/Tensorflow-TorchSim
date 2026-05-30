cc_library(
    name = "gtl_libs",
    srcs = glob([
        "*.cc",
    ]),
    hdrs = glob([
        "*.h",
    ]),
    linkstatic = 1,
    visibility = ["//visibility:public"],
    deps = [
        "//my_device/src/utils:prefetch",
        "//my_device/src/utils:types",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ],
)
