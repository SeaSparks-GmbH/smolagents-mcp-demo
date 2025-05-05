# Minimal Smolagents MCP Demo

This repository demonstrates how to use the **Model Context Protocol (MCP)** with multiple independent servers, a custom client, and an LLM-based host — all without relying on higher-level libraries like `FastMCP` but using lower-level methods of the MCP Python SDK.

It accompanies the Medium article:

**[MCP From a Slightly Different Angle — How to Build A Local Multi-Server System Without FastMCP](https://medium.com/@your-handle/your-article-slug)**

---

## Setup

1. Clone this repo  
2. (Recommended) Create a virtual environment  
3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Start the Servers

Either use the "Start server" cells provided in the notebook (`host.ipynb`) or start each server in a separate terminal:

```bash
python math_server.py
python weather_server.py
```

By default:
- MathServer runs on port `58000`
- WeatherServer runs on port `58001`

To use different ports, update the `.py` files accordingly.

---

## Use the Host

Open `host.ipynb` in Jupyter, execute the cells and try, e.g.:

```python
await run_host_query("How many hours are in 5 days?")
await run_host_query("What's the weather like in Berlin?")
```

The LLM will select the correct tool and the host will return the result from the tool call using MCP.

---

## Contents

- `math_server.py`: MCP server with `add` and `multiply` tools  
- `weather_server.py`: MCP server with `get_weather` tool that provides fake weather information
- `client.py`: Client that connects to multiple servers  
- `host.ipynb`: Interactive host logic using OpenAI  
- `requirements.txt`: Dependencies

---

## Contact

For questions or feedback, feel free to reach out:
[Email us](mailto:kontakt@seasparks.de)

## License

This repository is licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.
You are free to share and adapt the contents of this repository for non-commercial purposes, provided that appropriate credit is given.
For full details, see the LICENSE file.

**Note:** This license applies only to the original code and content included in this repository.
It does not apply to third-party libraries or tools installed at runtime,
which are governed by their own respective licenses.
