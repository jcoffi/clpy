import locale
import os
import subprocess
import tempfile


headercvt_abspath = os.path.join(
    os.path.dirname(__file__),
    "..",  # opencl_tests
    "..",  # clpy_tests
    "..",  # tests
    "..",  # clpy
    "headercvt",
    "headercvt")


def check_existence_of_headercvt():
    if not os.path.isfile(headercvt_abspath):
        raise FileNotFoundError("headercvt does not exist")


def exec_headercvt(workingdir, source):
    subprocess.run(
        [headercvt_abspath, "/dev/stdin", "--"],
        cwd=workingdir,
        input=source.encode(locale.getpreferredencoding()),
        timeout=1,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True
    )


def get_result_files(workingdir):
    with open(os.path.join(workingdir, "func_decl.pxi"), "r") as f:
        func_decl_str = f.read()
    with open(os.path.join(workingdir, "preprocessor_defines.pxi"), "r") as f:
        preprocessor_defines_str = f.read()
    with open(os.path.join(workingdir, "types.pxi"), "r") as f:
        types_str = f.read()
    return {
        "func_decl": func_decl_str,
        "preprocessor_defines": preprocessor_defines_str,
        "types": types_str
    }


def kick_headercvt_and_get_results(workingdir, source):
    exec_headercvt(workingdir, source)
    return get_result_files(workingdir)


def contains(result_string, match_string):
    return match_string in result_string


def compile_with(workingdir, source):
    source = """
include "func_decl.pxi"
include "preprocessor_defines.pxi"
include "types.pxi"


""" + source

    with open(os.path.join(workingdir, "test_case.pyx"), "w") as f:
        f.write(source)
        f.flush()
        os.fsync(f.fileno())
        try:
            subprocess.run(
                ["cython", f.name],
                cwd=workingdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(e.stdout.decode(locale.getpreferredencoding()))
            print(e.stderr.decode(locale.getpreferredencoding()))
            return False


def with_temp_wd(function):
    def impl(self):
        with tempfile.TemporaryDirectory() as wd:
            function(self, wd=wd)
    return impl
