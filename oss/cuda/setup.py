import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

NVCC_FLAGS = [
    "-O3", "-std=c++17",
    "--expt-relaxed-constexpr", "--expt-extended-lambda",
    "-lineinfo",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
    "-gencode=arch=compute_90,code=compute_90",
]

CUDA_HOME = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
INCLUDE_DIRS = []
if CUDA_HOME:
    conda_target_include = os.path.join(CUDA_HOME, "targets", "x86_64-linux", "include")
    if os.path.isdir(conda_target_include):
        INCLUDE_DIRS.append(conda_target_include)
    conda_cccl_include = os.path.join(conda_target_include, "cccl")
    if os.path.isdir(conda_cccl_include):
        INCLUDE_DIRS.append(conda_cccl_include)

setup(
    name="oss_cuda",
    version="0.2.0+phase2b",
    packages=["oss_cuda"],
    ext_modules=[
        CUDAExtension(
            name="oss_cuda._C",
            sources=["src/bindings.cpp", "src/rasterizer_fwd.cu"],
            include_dirs=INCLUDE_DIRS,
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17", "-fPIC"],
                "nvcc": NVCC_FLAGS,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
