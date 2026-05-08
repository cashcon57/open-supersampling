from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name="oss_cuda",
    version="0.1.0+phase1",
    packages=["oss_cuda"],
    ext_modules=[
        CppExtension(
            name="oss_cuda._C",
            sources=["src/bindings.cpp"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-fPIC"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
