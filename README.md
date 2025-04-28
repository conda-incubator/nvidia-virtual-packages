# nvidia-virtual-packages

A conda plugin which creates NVIDIA-specific virtual packages

The `__cuda_arch` virtual package provides the minimum compute capability of the available
CUDA devices detected on the system and the model description of the same device. This
virtual package may be used to enforce a minimum compute capability for a conda package or
build multiple variants of a conda package which each target one or a subset of CUDA
devices.

## Implementing a conda-recipe which depends on `__cuda_arch`

Define a `conda_build_config.yaml` to configure conda-build to build the recipe multiple
times. This file will need variables providing the compiler flags, compute capabilities, and
priority for each package variant.

In this example, we assume the build system is using CMake, so setting the `CUDAARCHS`
environment variable will tell CMake which compute capabilities to target.

In this example, we have three variants. One variant is built for the major versions 5
through 7, so it should be able to run on any device with compute capability `>=5,<8`. One
variant is built for compute capability 8.2 only. One variant is built for compute
capability 7.0, but it includes PTX, so it can run on any device with higher computer
capability as well.

> [!IMPORTANT]
> All packages should declare compatibility with `__cuda_arch=0` so that the packages may be
> installed into the test environment of a GPU-less build runner.

In this example, we have ranked the priority of the variants from most specific to least
specific so that users get the most optimized code for their device. This example is a bit
contrived, and it's probably not good practice to have multiple variants which are
compatible with a device. Priority is not needed if only one variant is compatible with
every possible compute capability.

```yaml
# conda_build_config.yaml

# CUDAARCHS is a CMake-specific environment variable
CUDAARCHS:
  - "50-real;60-real;70-real"
  - "82-real"
  - "70-real;70-virtual"

# Just for illustration, the equivalent args for pytorch would be
TORCH_CUDA_ARCH_LIST:
  - "5.0 6.0 7.0"
  - "8.2"
  - "7.0+PTX"

# These strings define the corresponding compatible compute capabilities
__cuda_arch:
  - "0 | >=5,<8"
  - "0 | 8.2.*"
  - "0 | >=7"

# We should rank the variants in case multiple variants match a user's machine
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

Finally, when building these package, we must set `CONDA_OVERRIDE_CUDA_ARCH="0"` so that our
build runner can test all variants of the package.
