# Minimal Smolagents MCP Demo

This repository demonstrates how to use HuggingFace’s lightweight Smolagents framework together with a custom Model Context Protocol (MCP) server — without relying on higher-level abstractions like `FastMCP`.

It features:
- A self-documenting MCP server exposing tools for a document library
- A Smolagents `MCPClient`, connecting seamlessly to the server
- A minimal `CodeAgent`-based MCP host, powered by an LLM, that calls tools, interprets outputs, and solves user queries through iterative Python code execution

The server components rely on the low-level MCP Python SDK to stay close to the protocol itself and remain fully transparent.
It accompanies the Medium article:
**[Taking MCP for a Spin, the Smol Way]()**

---

## Setup

1. Clone this repo  
2. (Recommended) Create a virtual environment  
3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Start the Server

Either use the "Start server" cell provided in the notebook (`host.ipynb`) or start the server in a separate terminal:

```bash
python library_server.py
```

By default:
- DocumentServer runs on port `58002`

To use a different port, update the `.py` files and notebook cells accordingly.

---

## Use the Host

Open `host.ipynb` in Jupyter, execute the cells and try, e.g.:

```python
run_host_query("Which documents are available?")
run_host_query("Give me a summary of the quantum squirrel report, please.")
```

The agent will automatically select the appropriate tool, using MCP under the hood, and on this basis answer the query.

---

## Contents

- `library_server.py`: MCP document server with `get_library` and `get_document` tools
- `build_mcp_tool_schema.py`: Helper function to create MCP tool metadata from docstring-based definitions
- `host.ipynb`: Interactive agent host logic using Smolagents and OpenAI
- `requirements.txt`: Dependencies
- `library/`: Directory with the documents to be served by the library server

---

## License and Usage

This repository is licensed under the [Creative Commons BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).  
It is intended for educational and non-commercial use.

**Note:** Dependencies (listed in `requirements.txt`) are used via pip and are **not redistributed** in this repository.  
Each retains its original license, typically Apache 2.0, MIT, BSD, or AGPL (e.g., `pymupdf`).  
If you plan to use this repository or its components in a commercial setting, please consult the respective licenses.

This license applies only to the original code and content included in this repository.
It does not apply to third-party libraries or tools installed at runtime,
which are governed by their own respective licenses.

---

## Contact

For questions or feedback, feel free to reach out:
[Email us](mailto:kontakt@seasparks.de)
