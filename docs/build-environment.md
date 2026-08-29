# Build environment (captured from WSL Ubuntu 24.04)

## Python
Python 3.12.3

## Compilers
/usr/bin/mpif90
/usr/bin/gfortran

Both report the same version -- `mpif90` here is an OpenMPI wrapper around
gfortran, not a separate compiler:

GNU Fortran (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Copyright (C) 2023 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


## Packages
ii  g++                              4:13.2.0-7ubuntu1                       amd64        GNU C++ compiler
ii  g++-13                           13.3.0-6ubuntu2~24.04.1                 amd64        GNU C++ compiler
ii  g++-13-x86-64-linux-gnu          13.3.0-6ubuntu2~24.04.1                 amd64        GNU C++ compiler for x86_64-linux-gnu architecture
ii  g++-x86-64-linux-gnu             4:13.2.0-7ubuntu1                       amd64        GNU C++ compiler for the amd64 architecture
ii  gcc                              4:13.2.0-7ubuntu1                       amd64        GNU C compiler
ii  gcc-13                           13.3.0-6ubuntu2~24.04.1                 amd64        GNU C compiler
ii  gcc-13-base:amd64                13.3.0-6ubuntu2~24.04.1                 amd64        GCC, the GNU Compiler Collection (base package)
ii  gcc-13-x86-64-linux-gnu          13.3.0-6ubuntu2~24.04.1                 amd64        GNU C compiler for the x86_64-linux-gnu architecture
ii  gcc-14-base:amd64                14.2.0-4ubuntu2~24.04.1                 amd64        GCC, the GNU Compiler Collection (base package)
ii  gcc-x86-64-linux-gnu             4:13.2.0-7ubuntu1                       amd64        GNU C compiler for the amd64 architecture
ii  gfortran                         4:13.2.0-7ubuntu1                       amd64        GNU Fortran 95 compiler
ii  gfortran-13                      13.3.0-6ubuntu2~24.04.1                 amd64        GNU Fortran compiler
ii  gfortran-13-x86-64-linux-gnu     13.3.0-6ubuntu2~24.04.1                 amd64        GNU Fortran compiler for the x86_64-linux-gnu architecture
ii  gfortran-x86-64-linux-gnu        4:13.2.0-7ubuntu1                       amd64        GNU Fortran 95 compiler for the amd64 architecture
ii  libblas3:amd64                   3.12.0-3build1.1                        amd64        Basic Linear Algebra Reference implementations, shared library
ii  libcaf-openmpi-3t64:amd64        2.10.2+ds-2.1build2                     amd64        Co-Array Fortran libraries  (OpenMPI)
ii  libcoarrays-openmpi-dev:amd64    2.10.2+ds-2.1build2                     amd64        Co-Array Fortran libraries  - development files (OpenMPI)
ii  libgfortran-13-dev:amd64         13.3.0-6ubuntu2~24.04.1                 amd64        Runtime library for GNU Fortran applications (development files)
ii  libgfortran5:amd64               14.2.0-4ubuntu2~24.04.1                 amd64        Runtime library for GNU Fortran applications
ii  liblapack3:amd64                 3.12.0-3build1.1                        amd64        Library of linear algebra routines 3 - shared version
ii  libllvm17t64:amd64               1:17.0.6-9ubuntu1                       amd64        Modular compiler and toolchain technologies, runtime library
ii  libllvm20:amd64                  1:20.1.2-0ubuntu1~24.04.3               amd64        Modular compiler and toolchain technologies, runtime library
ii  libmagic-mgc                     1:5.45-3build1                          amd64        File type determination library using "magic" numbers (compiled magic file)
ii  libmetis-dev                     5.1.0.dfsg-7build3                      amd64        Serial Graph Partitioning and Fill-reducing Matrix Ordering. Header
ii  libmetis5:amd64                  5.1.0.dfsg-7build3                      amd64        Serial Graph Partitioning and Fill-reducing Matrix Ordering
ii  libmumps-5.6t64:amd64            5.6.2-2.1build2                         amd64        Direct linear systems solver - parallel shared libraries
ii  libmumps-dev:amd64               5.6.2-2.1build2                         amd64        Direct linear systems solver - parallel development files
ii  libmumps-headers-dev             5.6.2-2.1build2                         all          Direct linear systems solver - development header files
ii  libopenblas-dev:amd64            0.3.26+ds-1ubuntu0.1                    amd64        Optimized BLAS (linear algebra) library (dev, meta)
ii  libopenblas-pthread-dev:amd64    0.3.26+ds-1ubuntu0.1                    amd64        Optimized BLAS (linear algebra) library (dev, pthread)
ii  libopenblas0:amd64               0.3.26+ds-1ubuntu0.1                    amd64        Optimized BLAS (linear algebra) library (meta)
ii  libopenblas0-pthread:amd64       0.3.26+ds-1ubuntu0.1                    amd64        Optimized BLAS (linear algebra) library (shared lib, pthread)
ii  libopenmpi-dev:amd64             4.1.6-7ubuntu2                          amd64        high performance message passing library -- header files
ii  libopenmpi3t64:amd64             4.1.6-7ubuntu2                          amd64        high performance message passing library -- shared library
ii  libscalapack-mpi-dev:amd64       2.2.1-3.1ubuntu1                        amd64        Scalable Linear Algebra Package - Dev files for MPI
ii  libscalapack-openmpi-dev:amd64   2.2.1-3.1ubuntu1                        amd64        Scalable Linear Algebra Package - Dev files for OpenMPI
ii  libscalapack-openmpi2.2:amd64    2.2.1-3.1ubuntu1                        amd64        Scalable Linear Algebra Package - Shared libs for OpenMPI
ii  libxkbcommon0:amd64              1.6.0-1build1                           amd64        library interface to the XKB compiler - shared library
ii  make                             4.3-4.1build2                           amd64        utility for directing compilation
ii  mpi-default-bin                  1.15build1                              amd64        Standard MPI runtime programs (metapackage)
ii  mpi-default-dev                  1.15build1                              amd64        Standard MPI development files (metapackage)
ii  openmpi-bin                      4.1.6-7ubuntu2                          amd64        high performance message passing library -- binaries
ii  openmpi-common                   4.1.6-7ubuntu2                          all          high performance message passing library -- common files
ii  rpcsvc-proto                     1.4.2-0ubuntu7                          amd64        RPC protocol compiler and definitions

## Compile command (WSL Ubuntu 24.04)

Run from inside `app/`:

    mpifort -fopenmp -I/usr/include tdgl.F90 -o tdgl \
        -ldmumps -lscalapack-openmpi -lopenblas

Two notes on that line:

- `.F90` (capital F) is preprocessed automatically, so the `-cpp` flag the
  original `.f95` build needed is no longer required.
- There is no `-O2`. gfortran defaults to `-O0`, so every run produced with
  this command was unoptimized.

Then:

    cd app  &&  mkdir -p data  &&  ./tdgl
    cd ../analysis  &&  python3 diagnostic.py

To confirm what actually got linked:

    ldd app/tdgl | grep -E 'mumps|scalapack|openblas|mpi'

Verified output of that build (2026-08-29):

    libdmumps-5.6.so         => /lib/x86_64-linux-gnu/libdmumps-5.6.so
    libmumps_common-5.6.so   => /lib/x86_64-linux-gnu/libmumps_common-5.6.so
    libscalapack-openmpi.so.2.2 => /lib/x86_64-linux-gnu/libscalapack-openmpi.so.2.2
    libopenblas.so.0         => /lib/x86_64-linux-gnu/libopenblas.so.0
    libmpi.so.40             => /lib/x86_64-linux-gnu/libmpi.so.40
    libmpi_mpifh.so.40       => /lib/x86_64-linux-gnu/libmpi_mpifh.so.40

i.e. the distro packages listed above, resolved exactly as intended.

## Cluster build (LSU, mike2)

The SLURM scripts under `sweeps/` do **not** use the toolchain above. Each one
loads:

    module load intel/2021.5.0 intel-mpi/2021.5.1

which gives:

| Component | Version |
|---|---|
| ifort (via `mpiifort`) | 2021.5.0 20211109 |
| Intel MPI | 2021.5.1 |
| MKL (`$MKLROOT`) | /home/packages/compilers/intel/mkl/2022.0.2 |

So the sweep binaries in `/work/idden/scripts/` were compiled with `mpiifort`,
and BLAS / LAPACK / ScaLAPACK come from MKL rather than the OpenBLAS and
ScaLAPACK packages listed above. Everything in the Packages section of this
file describes the WSL build only; `-lscalapack-openmpi -lopenblas` is not the
right link line on the cluster.

### Still unrecorded

The exact `mpiifort` link line, and where MUMPS comes from on the cluster --
MKL does not ship MUMPS, so it is either a separate module or a local build.
Both can be recovered from a binary that already exists:

    ls -l /work/idden/scripts/
    ldd /work/idden/scripts/<binary> | grep -E 'mumps|mkl|scalapack|mpi'
    grep -a mpiifort ~/.bash_history
    module avail 2>&1 | grep -i mumps

The `ldd` line is the useful one: it names every library the working binary
resolved, which is enough to reconstruct the link command. (`grep -a` matters
-- the history file contains a stray byte that makes grep treat it as binary.)

## Reproducing this environment

    sudo apt install -y gfortran make openmpi-bin libopenmpi-dev \
        libmumps-dev libmumps-headers-dev libscalapack-openmpi-dev \
        libopenblas-dev libmetis-dev
