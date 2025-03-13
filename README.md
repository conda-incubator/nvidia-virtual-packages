# nvidia-virtual-packages

A conda plugin which creates NVIDIA-specific virtual packages

The `__cuda_arch` virtual package provides the minimum compute capability of the available
CUDA devices detected on the system and the model description of the same device. This
virtual package may be used to enforce a mininmum compute capability for a conda package or
build multiple variants of a conda package which each target one or a subset of CUDA
devices.

## Implementing a conda-recipe which depends on `__cuda_arch`

Define a file namedd `conda_build_config.yaml` to tell conda-build to build the recipe
multiple times. This file will need variables which provide the compiler flags, compute
capabilities, and priority of each package variant.

In this example, we assume the build system is using CMake, so setting the `CUDAARCHS`
environment variable will tell CMake which compute capabilities to target.

```yaml
# conda_build_config.yaml

# CUDAARCHS is a CMake-specific environment variable
CUDAARCHS:
  - "50-real;60-real;70-real"
  - "82-real"
  - "80-real;80-virtual"

# Just for illustration, the equivalent args for pytorch would be
TORCH_CUDA_ARCH_LIST:
  - "5.0 6.0 7.0"
  - "8.2"
  - "8.0+PTX"

# These strings define the corresponding compatible compute capabilities
__cuda_arch:
  - ">=5.0,<8.0"
  - "8.2"
  - ">=8.0"

# We should rank the variants in case multiple variants are valid on a user's machine
priority:
  - 0
  - 2
  - 1

zip_keys:
  - __cuda_arch
  - CUDAARCHS
  - priority
```

In the recipe, we need to augment the build number according to install priority, pass the
compiler flags to the build environment as an environment variable, and set the
`__cuda_arch` package as run and host dependencies.

```yaml
# meta.yaml

{% set build = 0 %}

build:
  # Prioritize the build variants by increasing build number in-case there are multiple
  # valid matches
  number: {{ build + priority * 100 }}

env:
  # CUDAARCHS is an environment variable that CMAKE monitors to pass targets archs to
  # NVCC. We must mention all of our variant variables or else conda-smithy will strip
  # them out of the build matrix.
  - CUDAARCHS={{ CUDAARCHS }}

requirements:
  build:
    - {{ compiler('c') }}
    - {{ compiler('cxx') }}
    - {{ compiler('cuda') }}
    - {{ stdlib('c') }}
  host:
  # FIXME: Use a metapackage to implement strong run_exports instead?
  # TODO: Does it really matter if the host environment has the same __cuda_arch?
    - __cuda_arch {{ __cuda_arch }}
  run:
    - __cuda_arch {{ __cuda_arch }}

```
