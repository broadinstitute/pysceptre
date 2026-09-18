# Reproducible environment for the pysceptre vs R sceptre comparison.
#
# Exists because the two implementations were being measured on a host whose
# BLAS silently changes the result: numpy on macOS links Apple Accelerate,
# which threadpoolctl cannot introspect, so glm/irls.py's deliberate
# single-threading is a no-op there while it takes effect on Linux/OpenBLAS.
# Benchmarks taken on the two platforms exercise different code paths.
#
# This image pins Linux, OpenBLAS, R, sceptre, Python and thread counts, so a
# reported number can be reproduced rather than taken on trust.
#
# Build:  docker build -f docker/Dockerfile -t pysceptre-bench .
# Run:    docker/run_benchmark.sh   (sets --cpus and --memory)

FROM rocker/r-ver:4.5.1

# Thread counts are pinned here rather than left to the runtime: every
# comparison in the paper is single-core, and an unpinned OpenBLAS would
# quietly use every core it can see.
ENV OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMBA_NUM_THREADS=1 \
    R_LIBS_USER=/usr/local/lib/R/site-library \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gfortran \
        libcurl4-openssl-dev libssl-dev libxml2-dev libhdf5-dev zlib1g-dev \
        libopenblas-dev liblapack-dev \
        python3 python3-pip python3-venv python3-dev \
        git ca-certificates time \
    && rm -rf /var/lib/apt/lists/*

# --- R side -------------------------------------------------------------
# sceptre and ondisc are not on CRAN; pin by commit so the reference
# implementation cannot drift under the benchmark.
ARG SCEPTRE_REF=v0.10.3
ARG ONDISC_REF=main
RUN R -q -e "install.packages(c('remotes','Matrix','arrow','jsonlite'), repos='https://cloud.r-project.org')" \
 && R -q -e "remotes::install_github('timothy-barry/ondisc', ref='${ONDISC_REF}', upgrade='never')" \
 && R -q -e "remotes::install_github('Katsevich-Lab/sceptre', ref='${SCEPTRE_REF}', upgrade='never')" \
 && R -q -e "cat('sceptre', as.character(packageVersion('sceptre')), '| ondisc', as.character(packageVersion('ondisc')), '\n')"

# --- Python side --------------------------------------------------------
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY pyproject.toml README.md /src/
COPY src /src/src
RUN pip install --upgrade pip setuptools wheel \
 && pip install -e "/src[dev,fast,io]" \
 && python -c "import numpy, threadpoolctl; print('numpy', numpy.__version__); print(threadpoolctl.threadpool_info())"

COPY scripts /src/scripts
WORKDIR /src

# Record the environment so a result can be traced back to it.
RUN { echo "# environment recorded at image build"; \
      R -q -e "cat('R', as.character(getRversion()), '\n'); cat('sceptre', as.character(packageVersion('sceptre')), '\n'); si <- sessionInfo(); cat('BLAS', si\$BLAS, '\n')" 2>/dev/null | grep -v '^>'; \
      python -c "import sys, numpy, threadpoolctl; print('python', sys.version.split()[0]); print('numpy', numpy.__version__); [print('blas', p['internal_api'], p['num_threads']) for p in threadpoolctl.threadpool_info()]"; \
    } > /src/ENVIRONMENT.txt 2>&1 || true

CMD ["bash"]
