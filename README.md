CUDAkernelprintf
================

Enables near real-time and synchronous printf in CUDA kernels on older devices.

A CUDA kernel-level printf-implementation

This is a simple implementation of (nearly) synchronous printf on CUDA. It is able
to use STDOUT and STDERR, as well as printing to a log file. The usage is very similar
printf, but this implementation only supports the %d, %u, %f, %e, %c, and %s formating.

Since NVIDIA removed the device emulation capability from the CUDA SDK. Instead of device emulation, NVIDIA included a kernel level printf function.
However, the kernel-level printf requires a GPU with compute capability of at least 2.0. This leaves developers using older devices (like my laptop)
without a convenient debugging technique.

Furthermore, the NVIDIA kernel-level printf-implementation prints all messages as soon as the kernel exits. This is inconvenient for debugging
either long running GPU kernels or GPU kernels that mey get stuck.

While being very limited, this kernel-level printf-implementation addresses both of theses shortcomings. It works on devices with compute capaility 1.2
(and later) and is nearly synchronous.

CUDA device printf Since NVIDIA removed the device emulation from their CUDA
toolkit with the release of CUDA 3.1 NVIDIA is providing a kernel level printf
as a replacement. While this works nicely, there are some inherent limitations.
The printed strings are buffered in a fixed size buffer on the device and
printed to screen on the host as soon as the next host-level synchronisation
point, generally when the kernel finishes. Furthermore, the bufer is not
flushed when your programm exits. This leaves the problem how to debug kernel
that got stuck, like in an infinite loop, or simple iterate many times. My
solution to this problem is to provide a printf style function that prints
formatted strings to a buffer shared with the host. On the host side, a thread
reads the buffer periodically and prints its contents to screen. The figure
below shows the overall layout of the mechanism. There are 256 bufers defined
in page-locked or pinned host memory. What is page-locked or pinned host
memory? Usually, memory on the host system is paged. This allows the virtual
address space to be much larger than the physical one at the price that a
memory page may be swapped out. Page-locked memory is exempt from ever being
swapped out. A block of page-locked host memory can be mapped into the device's
adress space and is thus acessible from both, the host and the device. We use
this mechanism to transfer data asynchronous in parallel to the running kernel
from the device to the host. Prior to launching our CUDA kernel, a POSIX thread
(pthread) "writerThread" is launched to read the buffer periodically and writes
the contents to screen.


This software is free and open-source (released under the GPL V3).


Requirements:
The software has been developed using CUDA 3.2 on a Linux system. It should
work on other operating systems and earlier CUDA versions. However, adaptations
in the host code flushing the GPU buffers to screen or file may be needed.

A CUDA capable GPU with compute capabilities 1.2 or later, the NVIDIA CUDA
Toolkit and SDK, as well as the POSIX Thread library (pthreads) is required.


Usage:
Please see test.cu for a simple example.


Limitations:
i) Currently only understands the %d, %u, %f, %e, %c, and %s formats. The total length
of each string to be printed is 4095 chars. The number of buffers is limited to 256.
Buffer 0 is linked to stdout and buffer 1 to stderr, respectively, leaving 253
buffers available to be linked to any user defined stream.

ii) The number of formats is limited to 10. This is easily extendable via templates.

iii) While each call of dprintf is atomic, there is no guarantee that two consecutive
calls within a thread are not interrupted by any other thread.

iv) Technically, we only parse positive integer. So the smallest possible negative integer
is -(2^63 -1) instead of -2^63

v) The precision of floating point parsing is about 6 decimal places (on a device with
compute capability 1.2).

