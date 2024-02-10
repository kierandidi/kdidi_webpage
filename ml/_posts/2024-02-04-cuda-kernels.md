---
layout: post
title: How to use CUDA for PyTorch
image: /assets/img/blog/ssh_gpu/gpucluster.jpg
accent_image: 
  background: url('/assets/img/blog/jj-ying.jpg') center/cover
  overlay: false
accent_color: '#ccc'
theme_color: '#ccc'
description: >
  A bit of background on GPU acceleration and how to use it with PyTorch
invert_sidebar: true
categories: ml
---

# How to use CUDA for PyTorch

Recently the [CUDA MODE](https://www.youtube.com/@CUDAMODE) lecture series started with some amazing talks about how you can use tools like CUDA or Triton to speed up your PyTorch programs (join the [Discord](https://discord.com/invite/XsdDHGtk9N) in case you are interested to learn more). Here I want to summarise and review some of the concepts and tools from the lecture and write them together in a coherent blog post.

## 1. Profiling

Profiling is the process of measuring the time and resources that a program uses. It is a crucial step in the development of any software, as it allows you to identify bottlenecks and areas for improvement. In the context of GPU programming, profiling is especially important, as the performance of a GPU program can be highly dependent on factors such as memory access patterns, kernel launch configurations, and the specific hardware being used. It is also not trivial to profile GPU code, as the operations are executed asynchronously on the GPU and we cannot simply measure execution time like we would with CPU code. In the following sections are a few tools to get you started on that for PyTorch code (for this you need to have access to a GPU, e.g. via Google Colab or a local machine with a CUDA-enabled GPU).

### 1.1 Use `torch.cuda.Event`

To profile the time a torch opertion takes, you can use `torch.cuda.Event`. We cannot use the `time` module for this, because the operations are executed asynchronously on the GPU. Let us write a short function to profile the time a function call takes:

```python
import torch
def time_pytorch_function(func, input):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Warmup
    for _ in range(5):
        func(input)
    start.record()
    func(input)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)
```
We do a few warmup steps at the start to make sure that things like memory allocation calls, PyTorch's JIT fuser and other things are not included in the timing. Then we record the start and end of the function call and synchronize the GPU to make sure that the timing is correct. For more details on these things see [this blog post](https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch).

Let's try this with a simple toy example:

```python
b = torch.randn(100000, 100000).cuda()

def square_2(a):
    return a * a

def square_3(a):
    return a ** 2

print(time_pytorch_function(torch.square, b))
print(time_pytorch_function(square_2, b))
print(time_pytorch_function(square_3, b))
#output:
# 3.2753279209136963
# 3.272671937942505
# 3.2755520343780518
```
We can see that the multiplication `a * a` is slightly faster than the power operation `a ** 2`. However, we have no idea why this is happening; it is the same operation, so are they using different CUDA kernels? We can use the `torch.autograd.profiler` to find out.


### 1.2  Use `torch.autograd.profiler`

Fortunately, we do not have to write all profiling tools ourselves PyTorch has a built-in profiler. Let us look again at the same operations:

```python
print("=============")
print("Profiling torch.square")
print("=============")

# Now profile each function using pytorch profiler
with torch.autograd.profiler.profile(use_cuda=True) as prof:
    torch.square(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a * a")
print("=============")

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    square_2(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("=============")
print("Profiling a ** 2")
print("=============")

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    square_3(b)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

This gives us the following output:

![Profiling output](/assets/img/blog/gpu_profiling/simple_profiling.png)

We can see that `a * a` calls the faster `aten::mul` operation, while `a ** 2` calls the slower `aten::pow` operation, explaining our previous results.

ATen is a C++ library that is part of the [PyTorch C++ API](https://pytorch.org/cppdocs/). It is the foundational tensor and math library on which PyTorch is built and exposes the Tensor operations in PyTorch [directly in C++](https://pytorch.org/cppdocs/notes/tensor_basics.html). ATen is a very creative name, as it stands for "A tensor library". You can here more about the differences between the torch API and the ATen API in [this podcast episode](https://pytorch-dev-podcast.simplecast.com/episodes/torch-vs-aten-apis).
{:.note title="Aside"}

Let us now profile a simple neural network forward pass:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

data = torch.randn(1, 1, 32, 32).cuda()
with torch.autograd.profiler.profile(use_cuda=True) as prof:
  output = torch.nn.Linear(32, 32).cuda()(data)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

Which gives us the following output:

![Profiling output](/assets/img/blog/gpu_profiling/linear_profiling.png)

We can see that the `aten::linear` and the `aten::addmm` operation are the most time-consuming operations in this forward pass. In [another post]() I dig into how one can find the actual implementation of these functions in the PyTorch codebase to understand what they actually do, but for it is enough to know that `aten::linear` is the operation that applies a linear transformation to the input data and `aten::addmm` is the operation that performs a matrix multiplication of the input data with the weight matrix and adds a bias term.

### 1.3 Use `torch.profiler`

Another, more visual way to profile your code is to use `torch.profiler`. This is a more high-level interface to the profiler and allows you to export the profiling data to a Chrome trace file. Here is an example of how to use it:

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

def trace_handler(prof):
    print(prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,],    
    schedule=torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=2,
        repeat=1),
    on_trace_ready=trace_handler
    ) as p:
        for iter in range(10):
            output = torch.nn.Linear(32, 32).cuda()(torch.randn(1, 1, 32, 32).cuda())
            # send a signal to the profiler that the next iteration has started
            p.step()
```

We still get a terminal output:

![Profiling output](/assets/img/blog/gpu_profiling/pt_profiler.png)

However, we also get a Chrome trace file that we can open in Chrome to visualize the profiling data:

![Profiling output](/assets/img/blog/gpu_profiling/chrome_trace.png)

We can see that the majority of the time is actually spent on the cpu, moving data to the GPU, whereas the actual matrix multiplication is quite fast (and uses a special CUDA kernel called `volta_sgemm_32x32_sliced1x4_tn`).

### 1.4 Use `ncu` profiler

The `ncu` profiler is a command-line tool that comes with the CUDA toolkit. It is a very powerful tool that allows you to profile your CUDA kernels in great detail. You invoke it by running `ncu python script.py`. It will then run your script and profile all the CUDA kernels that are called. It will then generate a report in the form of a ncu_logs file that contains helpful numbers and recommendations on how to optimize your code.

![Profiling output](/assets/img/blog/gpu_profiling/ncu.png)

`ncu` also has a visual profiler that you can invoke by running `ncu --set full -o output $(which python) script.py`.

### 1.5 Use `Nsight` profiler

One can also use the [Nsight Systems] tool by NVIDIA, which gives a visual trace of memory usage and other performance metrics of your kernels.

## 2. Integrating CUDA kernels into PyTorch

CUDA is written in C++ and is a parallel computing platform and application programming interface (API) model created by Nvidia. It allows software developers to use a CUDA-enabled graphics processing unit (GPU) for general purpose processing.

### 2.1 `load_inline` function

The easiest way to integrate CUDA kernels into PyTorch is to use the `torch.utils.cpp_extension` module. This module allows you to compile C++ code into a shared library and then load it into Python. Here is an example of how to do this via the `load_inline` function for a simple matrix squaring operation:

```python
import torch
from torch.utils.cpp_extension import load_inline

# Define the CUDA kernel and C++ wrapper
cuda_source = '''
__global__ void square_matrix_kernel(const float* matrix, float* result, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int idx = row * width + col;
        result[idx] = matrix[idx] * matrix[idx];
    }
}

torch::Tensor square_matrix(torch::Tensor matrix) {
    const auto height = matrix.size(0);
    const auto width = matrix.size(1);

    auto result = torch::empty_like(matrix);

    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks((width + threads_per_block.x - 1) / threads_per_block.x,
                          (height + threads_per_block.y - 1) / threads_per_block.y);

    square_matrix_kernel<<<number_of_blocks, threads_per_block>>>(
        matrix.data_ptr<float>(), result.data_ptr<float>(), width, height);

    return result;
    }
'''

cpp_source = "torch::Tensor square_matrix(torch::Tensor matrix);"

# Load the CUDA kernel as a PyTorch extension
square_matrix_extension = load_inline(
    name='square_matrix_extension',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['square_matrix'],
    with_cuda=True,
    extra_cuda_cflags=["-O2"],
    # build_directory='./load_inline_cuda',
    # extra_cuda_cflags=['--expt-relaxed-constexpr']
)

a = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device='cuda')
print(square_matrix_extension.square_matrix(a))

# Output:
# tensor([[ 1.,  4.,  9.],
#         [16., 25., 36.]], device='cuda:0')
```

We see that the output is the same as if we used a PyTorch function. If we want to inspect the generated code, we can set the `build_directory` argument of the `load_inline` function to see the generated code in the specified directory.

### 2.2 Numba

Another way to integrate CUDA kernels into PyTorch is to use the `numba` library. This is a just-in-time (JIT) compiler that translates Python functions to optimized machine code at runtime using the industry-standard LLVM compiler library. It can also be used to generate CUDA kernels.

```python
from numba import cuda

# CUDA kernel
@cuda.jit
def square_matrix_kernel(matrix, result):
    # Calculate the row and column index for each thread
    row, col = cuda.grid(2)

    # Check if the thread's indices are within the bounds of the matrix
    if row < matrix.shape[0] and col < matrix.shape[1]:
        # Perform the square operation
        result[row, col] = matrix[row, col] ** 2

# Example usage
import numpy as np

# Create a sample matrix
matrix = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

# Allocate memory on the device
d_matrix = cuda.to_device(matrix)
d_result = cuda.device_array(matrix.shape, dtype=np.float32)

# Configure the blocks
threads_per_block = (16, 16)
blocks_per_grid_x = int(np.ceil(matrix.shape[0] / threads_per_block[0]))
blocks_per_grid_y = int(np.ceil(matrix.shape[1] / threads_per_block[1]))
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

# Launch the kernel
square_matrix_kernel[blocks_per_grid, threads_per_block](d_matrix, d_result)

# Copy the result back to the host
result = d_result.copy_to_host()

# Result is now in 'result' array
print(matrix)
print(result)
```


### 2.3 Triton

[Triton](https://openai.com/research/triton) is both a domain-specific language (DSL) and a compiler for writing highly efficient GPU code. It actually does not generate CUDA code, but PTX code, which is a lower-level intermediate representation of the CUDA code (basically the assembly language of CUDA). Newer features in PyTorch like `torch.compile` actually leverage Triton kernels under the hood, so it is worth understanding how it works. Since Triton is written in Python, it is easy to integrate with PyTorch. Here is an example of how to use Triton to write a simple matrix squaring operation:

```python
# Adapted straight from https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html
import triton
import triton.language as tl
import torch

@triton.jit
def square_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # The rows of the softmax are independent, so we parallelize across those
    row_idx = tl.program_id(0)
    # The stride represents how much we need to increase the pointer to advance 1 row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    # The block size is the next power of two greater than n_cols, so we can fit each
    # row in a single block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))

    square_output = row * row
    
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, square_output, mask=col_offsets < n_cols)


def square(x):
    n_rows, n_cols = x.shape
    # The block size is the smallest power of two greater than the number of columns in x
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps (num_warps) over which each row is distributed.
    # You will see in the next tutorial how to auto-tune this value in a more natural
    # way so you don't have to come up with manual heuristics yourself.
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    # Allocate output
    y = torch.empty_like(x)
    # Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row o
    # f the input matrix
    square_kernel[(n_rows, )](
        y,
        x,
        x.stride(0),
        y.stride(0),
        n_cols,
        num_warps=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


torch.manual_seed(0)
x = torch.randn(1823, 781, device='cuda')
y_triton = square(x)
y_torch = torch.square(x)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
```

We see that the output of the Triton kernel is the same as the output of the PyTorch function. 

Once we go to compiled code, we hopefully gain speed, but loose some of the flexibility that comes with eager execution, e.g. easy debugging via `pdb` and other Python debuggers or simple `print` statements.

Fortunately, Triton has a debugger now: we can invoke it by changing the `triton.jit` decorator to `triton.jit(interpret=True)`. This will allow you to set normal `Python` breakpoints and step through the kernel line by line. 

When doing that, you will see that most objects in the kernel are of the type `WrappedTensor`. So if you want to inspect a variable, you have to access its `.tensor` attribute.

To read more about Triton, you can have a look at the [original research paper](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf), a [video by the author Philippe Tillet](https://www.youtube.com/watch?v=G951lCm_qnk) and a [Reddit discussion](https://www.reddit.com/r/MachineLearning/comments/otdpkx/n_introducing_triton_opensource_gpu_programming/) where he himself gave some useful perspectives on the project.

What does Triton do under the hood? It converts the Python code first into a custom Triton IR and then via the Triton compiler into the well-known LLVM-IR. From there PTX code is generated. Basically, Triton leverages LLVM heavily and (quote from the paper) "just a few data- and control-flow extensions to LLVM-IR could enable various tile-level optimization passes which jointly lead to performance on-par with vendor libraries." These extensions allow Triton to do things like shared memory allocation or memory coalescence, things that in CUDA the GPU programmer has to handle manually.

![Triton under the hood](/assets/img/blog/gpu_profiling/triton.png)
From [this news article](https://international.binus.ac.id/computer-science/2022/09/02/openai-proposes-open-source-triton-language-as-an-alternative-to-nvidias-cuda/)

To get a feel for what Triton does under the hood, we can look at the PTX code generated for our Triton kernel. To do that, we can leverage the fact that `torch.compile` actually uses Triton under the hood. We can just write a simple function and then call `torch.compile` on it. Then, when running the script, we set the environment variable `TORCH_LOGS="output_code"` and run the script. This will create a directory `output_code` in the current working directory, which contains the Triton kernel.

```python
```

Looking at this, we can see that Triton leverages some heuristics to enable autotuning and other efficiency improvements. For example, it infers data types and element numbers and then uses this information to optimize the kernel.

One of the most important optimisations in ML is often kernel fusion, i.e. combining multiple operations into one kernel. It avoids the overhead of memory access and kernel launch.

We can also use the decorator `triton.testing.perf_report` to get a performance report of our kernel. 

```python
```


## Credits

Thanks to the [CUDA MODE](https://www.youtube.com/@CUDAMODE) lecture series for the inspiration for this post and the community around that for interesting discussions!

*[SERP]: Search Engine Results Page
