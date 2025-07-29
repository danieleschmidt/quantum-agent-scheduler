# Development container for Quantum Agent Scheduler
FROM mcr.microsoft.com/devcontainers/python:3.11-bullseye

# Install system dependencies
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        build-essential \
        curl \
        git \
        postgresql-client \
        redis-tools \
        vim \
        htop \
        jq \
        tree \
        unzip \
    && apt-get autoremove -y && apt-get clean -y

# Install Poetry
RUN pip install poetry==1.8.0 \
    && poetry config virtualenvs.create true \
    && poetry config virtualenvs.in-project true

# Install additional development tools
RUN pip install \
    ipython \
    jupyter \
    jupyterlab \
    notebook \
    ipywidgets

# Install quantum computing tools
RUN pip install \
    qiskit[visualization] \
    pennylane \
    cirq

# Configure Git (will be overridden by devcontainer)
RUN git config --global init.defaultBranch main

# Set up workspace
WORKDIR /workspace

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install Python dependencies
RUN poetry install --with dev,docs,quantum

# Create directories for development
RUN mkdir -p /workspace/logs /workspace/data /workspace/notebooks

# Set up Jupyter configuration
RUN jupyter notebook --generate-config \
    && echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py \
    && echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py

# Install VS Code extensions for better quantum development
RUN code-server --install-extension ms-python.python \
    && code-server --install-extension ms-toolsai.jupyter \
    && code-server --install-extension redhat.vscode-yaml \
    && code-server --install-extension charliermarsh.ruff

USER vscode