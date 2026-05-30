"""Setup TensorFlow as external dependency"""

_TF_HEADER_DIR = "TF_HEADER_DIR"

def _fail(msg):
    """Output failure message when auto configuration fails."""
    red = "\033[0;31m"
    no_color = "\033[0m"
    fail("%sPython Configuration Error:%s %s\n" % (red, no_color, msg))

def _tf_pip_impl(repository_ctx):
    # 1. 환경 변수 확인
    if _TF_HEADER_DIR not in repository_ctx.os.environ:
        _fail("'%s' environment variable is not set" % _TF_HEADER_DIR)
    
    tf_header_dir_str = repository_ctx.os.environ[_TF_HEADER_DIR]
    tf_header_path = repository_ctx.path(tf_header_dir_str)
    
    if not tf_header_path.exists:
        _fail("The path set in %s does not exist: %s" % (_TF_HEADER_DIR, tf_header_dir_str))

    # 2. 심볼릭 링크 생성
    # TF_HEADER_DIR 자체가 이미 'tensorflow/c/...' 구조를 가지고 있으므로
    # 이를 'include'라는 이름으로 통째로 연결합니다.
    repository_ctx.symlink(tf_header_path, "include")

    # 3. BUILD 파일 생성
    # includes = ["include"]를 추가하여 컴파일러가 'include' 폴더 내부를 찾게 합니다.
    repository_ctx.file("BUILD", """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "tf_header_lib",
    hdrs = glob(["include/**/*.h", "include/**/*.inc"], allow_empty = True),
    includes = ["include"],
)
    """)

tf_configure = repository_rule(
    environ = [
        _TF_HEADER_DIR,
    ],
    implementation = _tf_pip_impl,
)