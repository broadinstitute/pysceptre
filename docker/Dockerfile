# Minimal runtime image for pysceptre.
#
# Two stages: a builder that resolves and compiles the dependency tree, and a
# distroless runtime that keeps only the interpreter and the installed
# packages. No shell, no package manager, no apt, no build toolchain, and it
# runs as a non-root user.
#
# **The Python version is pinned by the runtime, not chosen.**
# `gcr.io/distroless/python3-debian12` ships Debian 12's interpreter, which is
# **3.11.2**, so the builder must be 3.11 as well. A 3.12 builder produces
# cp312 wheels for numpy and scipy that a 3.11 interpreter cannot import, and
# writes them to a `python3.12` site-packages directory that does not exist in
# the runtime. Bump both together or neither. 3.11 is inside the range
# `pyproject.toml` declares and CI verifies (3.10 to 3.13).
#
# **There is no thread pinning here**, unlike the benchmark image this
# replaced: forcing `OMP_NUM_THREADS=1` process-wide made single-core
# comparisons honest and would only slow a real analysis down. `glm/irls.py`
# already limits BLAS threading around the calls that need it, for reasons in
# `docs/design.md`.
#
# Build:  docker build -f docker/Dockerfile -t pysceptre .
#         docker build -f docker/Dockerfile --build-arg EXTRAS='[fast,io]' -t pysceptre .
# Run:    docker run --rm pysceptre -c "import pysceptre; print(pysceptre.__version__)"
#         docker run --rm -v "$PWD:/data" pysceptre /data/your_analysis.py
#
# The entrypoint is the interpreter itself, so arguments are Python's: a `-c`
# snippet, a `-m module`, or a mounted script path. pysceptre ships no CLI.

# ---------------------------------------------------------------- builder ---
FROM python:3.11-slim-bookworm AS builder
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# `numba` is the default extra, not an optional nicety: `crt/sampler.py` falls
# back to a pure-numpy grouping without it, and that grouping was the module's
# dominant cost at real scale. `io` is left out because nothing under `src/`
# imports it -- add it through EXTRAS when the container itself has to read an
# `.h5mu` or write parquet.
ARG EXTRAS="[fast]"

# README.md and both licence files are build inputs, not documentation:
# `pyproject.toml` names the first as `readme` and the other two in
# `license-files`, so the wheel cannot be built without them.
COPY pyproject.toml README.md LICENSE THIRD_PARTY_LICENSES ./
COPY src ./src

# No --compile-bytecode. It trades 93 MB of .pyc for a faster first import, and
# a container that runs a multi-minute analysis does not care about import
# latency. PYTHONDONTWRITEBYTECODE below keeps the runtime from writing any.
RUN uv pip install --no-cache ".${EXTRAS}"

# Slimming, measured. numpy, scipy and pandas each ship their own test package
# and a runtime image never runs them: 96 MB. Dropping compiled bytecode above
# is another 93 MB. Both figures are properties of the dependency set; the
# image total they produce also depends on the base image and the
# architecture, so it is not quoted here.
#
# **`strip` is deliberately not used here, and that is a measurement, not
# caution.** `strip --strip-unneeded` across the virtualenv's shared objects
# corrupted numpy's bundled OpenBLAS outright: `import numpy` then failed with
# "libscipy_openblas64_-7220728c.so: ELF load command address/offset not
# page-aligned". Restricting it to `llvmlite/binding/libllvmlite.so`, the
# largest single file in the image, does work and recovers only 23 MB of
# 168 MB, which is not a trade worth making against a JIT compiler's symbol
# table.
#
# The two bundled OpenBLAS copies (numpy.libs and scipy.libs, about 54 MB
# together) are left alone for the same family of reason: each wheel is built
# against its own, and deduplicating them is how you get a subtly wrong BLAS.
RUN find /opt/venv -type d -name tests -prune -exec rm -rf {} + \
 && find /opt/venv -type d -name __pycache__ -prune -exec rm -rf {} +

# ---------------------------------------------------------------- runtime ---
FROM gcr.io/distroless/python3-debian12

# Only the environment, never the source: the package is installed into the
# venv, so copying `src/` back in would ship a second, shadowing copy.
COPY --from=builder /opt/venv /opt/venv

# Distroless runs its own interpreter at /usr/bin/python3 rather than the
# venv's symlink, so the packages are exposed by path.
ENV PYTHONPATH="/opt/venv/lib/python3.11/site-packages" \
    PYTHONDONTWRITEBYTECODE=1

USER nonroot
