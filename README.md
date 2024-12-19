# Setting up a development environment

The dependencies in this project is managed via the `uv` tool. Step 0 is to [install
uv](https://docs.astral.sh/uv/getting-started/installation/) on your system. To install
the correct Python-version and all project dependencies, run:

```bash
uv sync
```

To activate the pre-commit hooks, run:

```bash
uv run pre-commit install
```

# Updating the retriever databases

The retriever databases are stored in the `output` directory. When updating, we will
read all the webcontents from https://olympiatoppen.no/ and https://olt-skala.nif.no/.
We will also index the content of some PDF files that are included as part of the
project.

```bash
uv run update-retrievers
```

# Start the webapp

The chainlit webapp listens to http://localhost:8888 by default. If you want it to
listen to another port, set the environmental variable `CHAINLIT_PORT`. To start the
chainlit webapp, run:

```bash
# Start the chainlit webapp on port 8888
uv run start-chainlit

# Optionally: Specify some other port
CHAINLIT_PORT=8910 uv run start-chainlit
```

You can then point your browser to http://localhost:8888 (or your port of choice).
