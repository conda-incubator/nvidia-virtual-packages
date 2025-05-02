# nvidia-virtual-packages

A conda plugin which creates NVIDIA-specific virtual packages

The `__cuda_arch` virtual package provides the **minimum** compute capability of the
available CUDA devices detected on the system and the model description of the same device.
This virtual package may be used to enforce a minimum compute capability for a conda package
or build multiple variants of a conda package which each target one or a subset of CUDA
devices.

## Implementing a conda-recipe which depends on `__cuda_arch`

Define a `conda_build_config.yaml` to configure conda-build to build the recipe multiple
times. This file will need variables providing the compiler flags, compute capabilities, and
priority for each package variant.

In this example, we assume the build system is using CMake, so setting the `CUDAARCHS`
environment variable will tell CMake which compute capabilities to target.

In this example, we have three variants. One variant is built for the major versions 5 and 6
with PTX for 6, so it should be able to run on any device with compute capability `>=5`. One
variant is built for compute capability 8.2 with PTX. One variant is built for compute
capability 7.0 with PTX.

> [!WARNING]
> Always include PTX/SASS with the highest targeted compute capability.
>
> Because the plugin detects only the **minimum** compute capability of the available CUDA
> devices on the system, there may be devices of higher compute capability on the system
> which may not be able to run the binary unless PTX/SASS is included.

In this example, we have ranked the priority of the variants from highest compute capability
to lowest compute capability so that users get the most complete instruction set for their
device.

```yaml
# conda_build_config.yaml

# CUDAARCHS is a CMake-specific environment variable
CUDAARCHS:
  - "82"
  - "70"
  - "50-real;60"

# Just for illustration, the equivalent args for pytorch would be
TORCH_CUDA_ARCH_LIST:
  - "8.2+PTX"
  - "7.0+PTX"
  - "5.0 6.0+PTX"

# These strings define the corresponding compatible compute capabilities
__cuda_arch:
  - ">=8.2"
  - ">=7"
  - ">=5"

# We should rank the variants in case multiple variants match a user's machine
# Higher numbers are higher priority
priority:
  - 2
  - 1
  - 0

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

Finally, when building these package, we must set `CONDA_OVERRIDE_CUDA_ARCH="999"` so that
our build runner can test all variants of the package.

## What about arch-specific- and family- instructions sets such as 90a and 120f?

If your program benefits from these instruction sets, use them! Every device that is `sm_90`
also supports the `sm_90a` instruction set, and every device that is `sm_120` also supports
the `sm_120f` instruction set. Thus, if this plugin returns `__cuda_arch=9.0`, then at least
one device on the system supports `sm_90a`.

However, since these instructions sets are not forward-compatible, so you should include
the non-specific/family instructions as SASS/PTX when the instruction set is the highest
target architecture.

For example, here were are targeting both family and specific instruction sets:

```yaml
CUDAARCHS:
  - "80-real;90a-real;100a-real;100f-real;100-virtual"

__cuda_arch:
  - "0 | >=8"
```

Notee that we have included `100-virtual` in order to provide forward-compatability.
`90-virtual` is not needed because any devices which `90-virtual` would run on also support
`90a-real` or `100-virtual`. Future devices may not support `100a-real` or `100f-real`, but
will support `100-virtual`.
