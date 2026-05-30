# Description:
#  Tools for building the TensorFlow pip package.

package(default_visibility = ["//visibility:private"])

load(
    "//my_device:demo_plugin.bzl",
    "transitive_hdrs",
)
load(
    "@local_config_syslibs//:build_defs.bzl",
    "if_not_system_lib",
)

# This returns a list of headers of all public header libraries (e.g.,
# framework, lib), and all of the transitive dependencies of those
# public headers.  Not all of the headers returned by the filegroup
# are public (e.g., internal headers that are included by public
# headers), but the internal headers need to be packaged in the
# pip_package for the public headers to be properly included.
#
# Public headers are therefore defined by those that are both:
#
# 1) "publicly visible" as defined by bazel
# 2) Have documentation.
#
# This matches the policy of "public" for our python API.

COMMON_PIP_DEPS = [
    "MANIFEST.in",
    "README",
    "setup.py",
    "//my_device:libdemo_plugin.so",
]

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = COMMON_PIP_DEPS,
)
