# nvidia-virtual-packages

A conda plugin which creates NVIDIA-specific virtual packages.

The `__cuda_arch` virtual package provides the **minimum** compute capability of the
available CUDA devices detected on the system. This virtual package may be used to enforce a
minimum compute capability for a conda package or build multiple variants of a conda package
which each target one or a subset of CUDA devices.

Similar to how the virtual package `__cuda` constrains the `cuda-version` metapackage, which
represents to conda the CUDA *driver* version available on the system, this plugin creates
the virtual package `__cuda_arch`, which constrains the `cuda-arch` metapackage, which
represents to conda the minimum compute capability of all CUDA *devices* on the system.

## The `cuda-arch` metapackage

Recipes and packages should not depend on a specific version of `__cuda` or `__cuda_arch`
directly because wrapper metapackages like `cuda-version` or `cuda-arch` can carry extra
behavior (like run_exports) which virtual packages cannot. For example, we can automatically
upgrade users to `cuda-version=13.4` even when conda reports `__cuda=13.0` because of CUDA
minor version compatibility. Deploying a metapackage also allows users to easily modify the
constraints of their environment without setting an environment variable such as
`CONDA_OVERRIDE_CUDA`. For example, you can request a `cuda-version=12.6` environment when
conda reports `__cuda=13.2` because of the CUDA driver's backward compatibility.

> [!IMPORTANT]
> This plugin does not create the `cuda-arch` metapackage. This package must be created
> separately and published to the channel.

The recipe for `cuda-arch` looks like this:

```yaml
package:
  name: cuda-arch
  version: ${{ cuda_arch_version }}

requirements:
  run_exports:
    - cuda-arch >=${{ cuda_arch_version }}
  run:
    - __cuda_arch >=${{ cuda_arch_version }}
```

The behavior encoded into the `cuda-arch` metapackage is that a package built with
`cuda-arch=6.1` will be installable on a system reporting `__cuda_arch>=6.1`. This mirrors the
forward compatibility of PTX.

One version is published for every known major-minor compute capability. Sub-architectures
such as `100f` are not expressible within this framework but can still be targeted at build
time.

## Implementing a conda recipe which depends on `cuda-arch`

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
> Always include PTX with the highest targeted compute capability.
>
> Because the plugin detects only the **minimum** compute capability of the available CUDA
> devices on the system, there may be devices of higher compute capability on the system
> which may not be able to run the binary unless PTX is included.

In this example, we have ranked the priority of the variants from highest compute capability
to lowest compute capability so that users get the most complete instruction set for their
device.

```yaml
# conda_build_config.yaml

# CUDAARCHS is a CMake-specific environment variable
CUDAARCHS:
  - "82"
  - "70-real;70-virtual"
  - "50-real;60"

# Just for illustration, the equivalent args for pytorch would be
TORCH_CUDA_ARCH_LIST:
  - "8.2+PTX"
  - "7.0+PTX"
  - "5.0 6.0+PTX"

# These strings define the corresponding compatible compute capabilities
cuda_arch_version:
  - "8.2"
  - "7.0"
  - "5.0"

# We should rank the variants in case multiple variants match a user's machine
# Higher numbers are higher priority
priority:
  - 2
  - 1
  - 0

zip_keys:
  -
    - cuda_arch_version
    - CUDAARCHS
    - priority
```

> [!IMPORTANT]
> The variant variable containing the minimum supported arch for each build MUST be named
> "cuda_arch_version" because `conda-forge-ci-setup` searches for this variable name
> when deciding what value to set `CONDA_OVERRIDE_CUDA_ARCH` to on the build runner.

In the recipe, we need to augment the build number according to install priority, pass the
compiler flags to the build environment as an environment variable, and set the
`cuda-arch` package as run and host dependencies.

```yaml
# meta.yaml

{% set build = 0 %}

build:
  # Prioritize the build variants by increasing build number in case there are multiple
  # valid matches
  number: {{ build + priority * 100 }}

env:
  # CUDAARCHS is an environment variable that CMake monitors to pass target archs to
  # NVCC. We must mention all of our variant variables or else conda-smithy may strip
  # them out of the build matrix.
  - CUDAARCHS={{ CUDAARCHS }}

requirements:
  build:
    - {{ compiler('c') }}
    - {{ compiler('cxx') }}
    - {{ compiler('cuda') }}
    - {{ stdlib('c') }}
  host:
  # We must pin cuda-arch in the host environment to the minimum supported cuda-arch to
  # ensure that dependencies are also compatible with the minimum supported cuda-arch
    - cuda-arch {{ cuda_arch_version }}

```

## What about arch-specific and family-specific instruction sets such as 90a and 120f?

If your program benefits from these instruction sets, use them! Every device that is `sm_90`
also supports the `sm_90a` instruction set, and every device that is `sm_120` also supports
the `sm_120f` instruction set. Thus, if this plugin returns `__cuda_arch=9.0`, then at least
one device on the system supports `sm_90a`.

However, since these instruction sets are not forward-compatible, you should include
the non-specific/family instructions as PTX when the instruction set is the highest
target architecture.

For example, here we are targeting both family and specific instruction sets:

```yaml
CUDAARCHS:
  - "80-real;90a-real;100a-real;100f-real;100-virtual"

cuda_arch_version:
  - "8.0"
```

Note that we have included `100-virtual` in order to provide forward-compatibility.
`90-virtual` is not needed because any devices which `90-virtual` would run on also support
`90a-real` or `100-virtual`. Future devices may not support `100a-real` or `100f-real`, but
will support `100-virtual`.
