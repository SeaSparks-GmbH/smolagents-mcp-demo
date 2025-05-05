# Taking MCP for a Spin, the Smol Way

## Introduction

In our previous article on this topic ([MCP From a Slightly Different Angle...](https://medium.com/@haegler/mcp-from-a-slightly-different-angle-how-to-build-a-local-multi-server-systems-without-fastmcp-495356880d13)), we explored the Model Context Protocol (MCP) from a practical and exploratory angle — building a multi-server local system *without* relying on FastMCP. That post aimed to make MCP more accessible by demonstrating how you can set up your own environment using the official MCP Python SDK ([Python SDK on GitHub](https://github.com/modelcontextprotocol/python-sdk)) and stay close to the protocol itself, thereby showing some of its internals that are otherwise somewhat obscured behind layers of abstraction.

This time, we’re going one step further.

Enter Smolagents, a lightweight agent framework from Hugging Face that does a lot with very little ([Smolagents on HF](https://huggingface.co/docs/smolagents/index))([on GitHub](https://github.com/huggingface/smolagents)). Compact and extensible, Smolagents encourages playing with agents while staying true to agentic principles. One of its standout features is the concept of Code Agents — agents that write and execute Python code in real time. This design turns a language model into an active "problem-solver with a keyboard", and plays to the strengths of many of the modern powerful LLMs, which often excel at writing Python. And even more importantly: agents that write and execute code are FUN.

In this article, we’ll combine the minimal yet powerful MCP system from last time, based on the MCP-trinity of Server, Client and Host, with the lightweight agent framework of Smolagents to demonstrate:

- A more complex MCP server setup (library_server.py), capable of exposing real functions in a self-documented way.
- How Smolagents' MCPClient can effortlessly integrate with this, letting agents connect to services as tools.
- How to wire it all together: setting up a minimal "MCP Host" with a Smolagent-driven workflow and running real examples.

By the end of this walkthrough, you’ll not only have a public repo to try out (link coming up), but also a deeper understanding of how Smolagents and MCP can work together to build elegant, distributed agent systems.

Smol spoiler: Our agent, accessing the library, will give you access to some one-of a kind hilarious, whisimcal little stories. Think of it as honey pot at the end of the rainbow / road ...

## A Realistic MCP Document Library Server - Code Walkthrough

In the previous article, we introduced a basic MCP setup with ubiquitous math server and fake weather server — good for learning the basics, but not exactly useful in a real-world agent workflow. Especially with Smolagents now in the mix, and particularly their CodeAgent, the old math server becomes... well, redundant. After all, a CodeAgent can simply write Python to add, multiply and *so much more* — no need to ask a remote server.

So in this article, we’re moving to a more realistic and functional MCP server: a document library server that actually gives agents something worth interacting with.

Let’s walk through it piece by piece. Many parts will feel familiar — but some new patterns will also emerge, especially in how tools are described and served.

### Imports and Library Setup

We’re using:

```python
import os
import json
from rapidfuzz import process, fuzz
import fitz # pip install pymupdf
from mcp.server.lowlevel import Server
from mcp.server.sse import SseServerTransport
from mcp.types import CallToolResult, TextContent
from starlette.applications import Starlette
from starlette.routing import Route, Mount
import uvicorn
from build_mcp_tool_schema import build_tool_schema

library_path = "./library"
```

New in this setup are in particular
- pymupdf (fitz) to extract text from PDFs.
- rapidfuzz for fuzzy title matching—useful when users (or agents!) get document titles almost right.
- build_tool_schema, a custom method (to be found in our git repo) that greatly simplifies the construction of the MCP Tool schema below.
- The library_path simply points to a local folder of .pdf files that represent the knowledge base, setting the stage for a useful content source.

### The MCP Entry Point

```python
server = Server("DocumentServer")

@server.call_tool()
async def call_tool(name: str, arguments: dict)-> CallToolResult:
    if name == "get_document":
        document_text = get_document(**arguments)

        return_value = TextContent(type="text", text=document_text)

        return [return_value]

    if name == "get_library":
        library_titles = get_library(**arguments)

        return_value = TextContent(type="text", text=library_titles)

        return [return_value]

    raise ValueError("Unknown tool")
```

This is the heart of the MCP Server. As covered in the last article, this function is called by the MCP Client when an agent invokes a tool. Frameworks like FastMCP or Smolagents' ToolHandler typically abstract this away, but here we keep full control.

Next, we take a brief look at the two tools that are handled: "get_document" and "get_library", returning plain text results wrapped in MCP’s TextContent format.


### Tool Logic
The simplest of the two just lists the available PDF filenames in the library. This helps not only agents but in particularly the user decide what to ask for:

```python
def get_library() -> str:
    """Gets a list of documents in the library.
       The output is of the form {"list_of_files":[file1, file2, ...]}

    Args:

    """

    pdf_files = [f for f in os.listdir(library_path) if f.lower().endswith(".pdf")]

    return json.dumps({"list_of_files": pdf_files})
```

Next, things get a bit smarter. Rather than expecting exact filename matches, we use fuzzy matching (token_set_ratio) to map approximate titles to the best available PDF.

```python
def get_document(title: str) -> str:
    """Gets the content of a document.

    Args:
         title: The title of the document.
    """

    pdf_files = json.loads(get_library())['list_of_files']
    pdf_files_cleaned = [f.lower().strip().replace("_", " ") for f in pdf_files]

    clean_title = title.lower().strip()

    # Best fuzzy match using token_set_ratio
    best_match, score, _ = process.extractOne(
        clean_title,
        pdf_files_cleaned,
        scorer=fuzz.token_set_ratio,
        score_cutoff=50
    )

    if not best_match:
        return f"No matching document found for: {title}"

    # Recover original filename
    matched_index = pdf_files_cleaned.index(best_match)
    file_path = os.path.join(library_path, pdf_files[matched_index])

    try:
        with fitz.open(file_path) as doc:
            text = "\n".join(page.get_text() for page in doc)
        return text.strip() or f"Couldn't extract the content of document '{title}'."
    except Exception as e:
        return f"Error reading document '{title}': {e}"
```

Once found, the file is opened and all its pages are extracted to plain text. The returned text can then be fed directly to the agent’s LLM context.

### Tool Description

```python
@server.list_tools()
async def list_tools():
    return [
        build_tool_schema(get_library),
        build_tool_schema(get_document),
    ]
```

This is new compared to our earlier setup. We now use a utility function `build_tool_schema(...)` to generate full MCP-compatible descriptions for our tools—including their names, descriptions and input arguments, based on their docstrings.

This simplifies tool registration and avoids manual schema construction. Again, frameworks like FastMCP often do this under the hood — but doing it yourself at least once explicitely like this helps understanding it on a different level.

### The Transport Layer

This final part handles the SSE-based communication layer, just like in our last article. Nothing major has changed here — if you want a deeper explanation, refer back to the transport section in Part 1.

```python
sse = SseServerTransport("/messages/")

async def sse_handler(request):
    async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())

routes = [
    Route("/sse", endpoint=sse_handler),
    Mount("/messages/", app=sse.handle_post_message),
]

app = Starlette(routes=routes)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=58002)
```

All of the above is brought together in `library_server.py` in the accompanying GitHub repo (...).

The server can be started as usual just with `python library_erver.py`.

## The MCP Client: Talking to the Server the Smol Way

With our DocumentServer up and running, we now turn to the agent side of the equation.

Instead of manually wiring up HTTP calls or relying on complex wrappers, Smolagents provides a minimal but effective MCPClient, which can connect to one or more MCP-compatible servers, fetch their tools, and return a clean, agent-ready interface.

Here’s a simple example of how it works:

```python
from smolagents.protocols.mcp_client import MCPClient

url_list = [{"url": "http://localhost:58002/sse"}]

def show_tools(url_list=url_list):
    try:
        mcp_client = MCPClient(url_list)
        tools = mcp_client.get_tools()
        for tool in tools:
            print("------------------------------------")
            print("tool.name ", tool.name)
            print("tool.description", tool.description)
            print("tool.inputs ", tool.inputs)
            print("tool.output_type", tool.output_type)
            print("------------------------------------")
    finally:
        mcp_client.disconnect()

show_tools()
```

Let us quickly walk through this, too:
We create a simple url_list pointing to our local MCP servers - feel free to add e.g. your math, weather etc servers running remote or locally under their respective ports.

Note that MCPClient accepts a *list* of servers and establishes sessions via Server-Sent Events (SSE). It discovers and retrieves all the tool definitions (as provided by @server.list_tools()) form all servers.
Now that the tools are ready and accesible, our agents can use them — or we test them first interactively, like in the above script.

Thats how a typical output looks like:

``` 
INFO:     127.0.0.1:56552 - "POST /messages/..." 202 Accepted
------------------------------------
tool.name:  get_library
tool.description:  Gets a list of documents in the library.
tool.inputs:  {}
tool.output_type:  string
------------------------------------
tool.name:  get_document
tool.description:  Gets the content of a document.
tool.inputs:  {'title': {'type': 'string', 'description': 'The title of the document.'}}
tool.output_type:  string
------------------------------------
```

You can see that the MCPClient gives you:

A full list of available tools (in this case: get_library and get_document)
Each tool’s name, description, input schema, and output type
No extra YAML, no hardcoded OpenAPI specs — just introspectable tools, as described by the server itself.

**Why This Matters**  
This example may look minimal, but it’s doing a lot:

- Server discovery: Find out what the server offers.
- Schema understanding: Parse tool arguments and return types.
- Agent preparation: This is all you need to attach tools to a Smolagent (coming next).

This is where simplicity becomes powerful. Whether you’re building a UI, wiring an agent, or testing toolchains, this tiny client handles the connection and lets you focus on what to do with the tools—which is exactly what we'll tackle next.

## The Host: A Super-Minimal Smolagent Setup

Now it’s time to bring the pieces together.

To keep things focused and beginner-friendly, we’ll set up just a single agent, using Smolagents’ wonderful `CodeAgent`. This agent can retrieve tools from the MCP server, generate Python code to use those tools, read and reason over results, and iterate until it finds a valid solution.

Yes, it's only *one* agent — but it's already doing a lot.

**Why CodeAgent?**  
Because most powerful LLMs are very good at writing and fixing Python code. With CodeAgent, the model is allowed to generate runnable Python, observe its output, handle errors, and retry as needed—until the task is done.

Even simple tasks can involve a lot of interaction and self-correction, which leads us to an important note…

**Watch your tokens!**  
Agentic calls are more expensive than standard one-shot LLM completions. Even without long documents, a single agent cycle might:
- Generate multi-step Python code
- Execute tools and parse outputs
- Read error messages and try again
- Repeat several times before producing a final response

So for this demo, we’ll use gpt-4o-mini to keep costs under control—while still getting sufficiently strong performance.

**Minimal Host Setup**  
Here’s the full host code:

```python
import os
from smolagents.agents import CodeAgent
from smolagents.models import OpenAIServerModel

openai_api_key = "your-openai-key"
load_params = {"temperature": 0.01, "max_tokens": 5000}

model = OpenAIServerModel(
    model_id="gpt-4o-mini",
    api_key=openai_api_key,
    **load_params
)

def run_host_query(user_input: str, url_list=url_list):
    with MCPClient(url_list) as tools:
        agent = CodeAgent(
            tools=tools,
            model=model,
            additional_authorized_imports=["json"]
        )
        return agent.run(user_input)
```

**What This Does**  
- Initializes an OpenAI-compatible model (swap in your own remote or local inference server if needed).
- Connects to our MCP server to fetch the tool list.
- Instantiates a `CodeAgent` that can use those tools and generate Python code to solve your task.
- Runs the agent on a given user_input.

This is about as minimal as it gets—but it already supports MCP tool use, Python execution, retry loops, and full reasoning.

In the final section, we’ll show some fun examples—real queries, real agent behavior, real results.

Stay tuned. It’s going to be worth it.

"The Quantum Squirrel report, written by Timothy J. Baxter, describes a book about Quince, a squirrel who appears normal but possesses quantum abilities, like existing in multiple places at once. The story involves Nate, a boy who traps Quince with peanut butter and later engages in deep conversations with him about quantum concepts and potential realities. Quince reveals visions of a future where squirrels have evolved into 'cognitively networked hunger forms.' The book is complex and mixes humor with philosophical themes. Timothy rates it four acorns out of five, noting it as confusing yet interesting."

"In the story 'The Turnip That Spoke Latin,' a poor farmer plants turnips in barren soil. One turnip sprouts and speaks Latin phrases, prompting the farmer to learn the language. The turnip shares existential thoughts, questioning the necessity of consuming it for survival, ultimately leading the farmer to recognize the value of wisdom over mere hunger. After an emotional struggle, the farmer eats the turnip, which ceases to speak, but the lessons learned from it remain with him, ensuring his fields are never empty again."

'\nThe document proposes a reinterpretation of dark energy as an emergent phenomenon from the consumption of black coffee, introducing a "Coffee Energy Conservation Law" linked to cognitive output stability and a concept called Mug Rotational Invariance (MRI). It argues that consuming coffee defines a closed energy system with symmetries akin to quantum theories. The introduction of cup handles disrupts these symmetries, causing localized excitations (sippinos) that can affect cognitive states. Experimental results suggest that drinking from handle-less mugs enhances attention and reduces errors. The conclusion emphasizes that productivity hinges not solely on caffeine but also on the geometric properties of the coffee mug.\n'

### MCP Server

An MCP server hosts tools. Each tool exposes a defined interface (name, parameters, description), and responds to standard MCP messages (like initialize, tools/list, tools/call). These servers can run locally, on remote machines, or be embedded in external systems. They are modular, stateless, and reusable: anything, anywhere.

### MCP Client

A client establishes the actual connection to one or more servers. It sends requests, handles responses, and often abstracts transport protocols (like stdin/stdout, SSE, or http-stream). Clients are typically “thin”: they act as low-level connectors between tools and the system (host or agent) that wants to use them.

### MCP Host

The host coordinates everything. It’s not a "host" in the traditional networking sense (like a physical machine), but rather the central component in an LLM system — typically the one talking to the LLM itself. It initializes client connections, discovers available tools, and formulates tool calls based on user needs or LLM decisions.

Think of it as the hub that enables the LLM to interact with the right tools at the right time.

Here’s a simplified overview of the architecture we'll build (up to the actual app / user interaction):

```
   ┌──────────────┐       ┌───────────────┐
   │  MathServer  │       │ WeatherServer │
   │  [add, mult] │       │ [get_weather] │
   └──────┬───────┘       └──────┬────────┘
          │                      │
   ┌──────▼──────────────────────▼──────┐
   │            MCP Client              │
   │ (connects to both servers, routes  │
   │  tool calls via common interface)  │
   └──────────────┬─────────────────────┘
                  │
         ┌────────▼─────────┐
         │      MCP Host    │
         │ (talks with LLM) │
         └────────┬─────────┘
                  │
               ┌──▼──┐
               │ App │
               └──┬──┘
                  │
                 User


Architecture of a minimal MCP multi-server setup
```

In the following sections, we'll implement this architecture using the lower-level mcp.server.lowlevel, mcp.client.session, and related classes from the Python MCP SDK. Rather than relying on high-level wrappers like fastmcp, we’ll stay closer to the metal to give you insight into how things actually work.

To keep things simple and focused, we’ll use tools from the Core Utilities & Assistants category — enough to demonstrate real, working multi-server setups without overwhelming complexity.

## Exemplary MCP Servers Using the Python MCP SDK

As mentioned earlier, we’ll use the lower-level MCP SDK classes directly — especially `Server` and `SseServerTransport`. This gives us full control over server behavior and lets us see exactly how a tool is made discoverable and callable via the Model Context Protocol.

> Everything shown below is based on **MCP SDK version 1.6.0**.  
> Install it with e.g.:
>
> ```bash
> pip install mcp==1.6.0
> ```

Additional libraries (like starlette, uvicorn, or openai) are used throughout the examples. These will need to be installed as well, though we won’t call out each one individually.

### 1. Basic Server Setup

To begin, we import and instantiate the server:

```python
from mcp.server.lowlevel import Server
server = Server("MathServer")
```

This registers a new server with the name `MathServer`.

### 2. Define Your Tool(s)

We define each tool using plain Python functions — no decorators needed. What matters is:

- Type hints for all parameters and return values.
- A clean docstring describing its behavior (this is helpful for downstream tool summaries).

```python
def add(a: int, b: int) -> int:
    """Adds two integers.

    Args:
         a: The first integer.
         b: The second integer.
    """
    return a + b
```

### 3. Tool Discovery

MCP needs a way to “see” which tools a server supports. This is done with a `@server.list_tools()` function that returns a list of `Tool` objects — each specifying name, description, input schema, etc.

```python
import mcp.types as types

@server.list_tools()
async def list_tools():
    return [
        types.Tool(
            name="add",
            description="Adds two integers",
            inputSchema={
                "type": "object",
                "required": ["a", "b"],
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"},
                },
            },
        ),
        types.Tool(
            name="multiply", 
            description="Multiplies two integers",
            inputSchema={ ... },
        ),
    ]
```

### 4. Tool Invocation

Now we need to define *how* the server will actually execute tool calls. This is done with the `@server.call_tool()` decorator:

```python
from mcp.types import TextContent

@server.call_tool()
async def handle_call(func_name: str, args: dict):
    if func_name not in ["add", "multiply"]:
        raise ValueError("Unknown tool")

    func = globals().get(func_name)
    if callable(func):
        result = func(**args)
        return [TextContent(type="text", text=str(result))]
    else:
        return [TextContent(type="text", text=f"Function {func_name} not defined.")]
```

> Note: We wrap the result in a list of `TextContent`, which is the expected format for returning tool output. Even if you only return a string, it must be inside a `TextContent` and placed in a list — otherwise, the call will fail.

### 5. Serving Over SSE with Starlette

To enable real-time communication, we expose our server over **Server-Sent Events (SSE)** using an ASGI-compatible app — here, we use [Starlette](https://www.starlette.io/) because it's lightweight and already integrated with MCP SDK classes.

> ⚠️ **Note:** SSE transport is now deprecated in favor of [Streamable HTTP](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/206), but still supported in MCP v1.6.0.

#### Define the transport:

```python
from mcp.server.sse import SseServerTransport

sse_transport = SseServerTransport("/messages/")
```

#### Route SSE requests to your server:

```python
async def sse_handler(request):
    async with sse_transport.connect_sse(request.scope, request.receive, request._send) as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())
```

#### Define Starlette routes:

```python
from starlette.applications import Starlette
from starlette.routing import Route, Mount

routes = [
    Route("/sse", endpoint=sse_handler),
    Mount("/messages/", app=sse_transport.handle_post_message),
]

starlette_app = Starlette(routes=routes)
```

### 6. Start the Server

Finally, launch the ASGI app with `uvicorn`, binding it to your chosen port (e.g. 58000):

```python
import uvicorn

if __name__ == "__main__":
    uvicorn.run(starlette_app, host="0.0.0.0", port=58000)
```

### 7. Run and Extend

You can now save the script as `math_server.py` and start it with:

```bash
python math_server.py
```

We also define a second server — `weather_server.py` — listening on a different port (e.g., 58001) and offering a fake weather tool. That way, we’ll be able to run multiple servers in parallel on the same machine (or across machines, if desired).

The full code — including both servers, client setup, and host logic — is available in our accompanying GitHub repository: [mcp-multi-server-demo](https://github.com/SeaSparks-GmbH/mcp-multi-server-demo).

## Exemplary Client

Now that the servers are running, we need to establish the "thin" client that allows the host to connect to the servers and their tools in a standardized way.

To handle the connection, sessions and communication with the servers, we use and start with setting up a MultiServerClient class, which will hold all methods needed to  
- initialize the connection to the servers  
- request a list of all the tools that are served by the servers  
- place calls to the tools from the host and communicate the results of the calls back to the host  
- close the sessions gracefully

The `__init__` just sets the stage for the connection to the servers (sse-endpoints) and the exit.

```python
from contextlib import AsyncExitStack

class MultiServerClient:
    def __init__(self, endpoints):
        """
        Initialize with a dictionary of {server_name: sse_url}.
        Example:
        {
            "math_server": "http://127.0.0.1:58000/sse",
            "weather_server": "http://127.0.0.1:58001/sse"
        }
        """
        self.endpoints = endpoints
        self.sessions = {}
        self._exit_stack = AsyncExitStack()
```

Next, we define compact methods within the class to connect to all servers (initialization via SSE) and to close them when the connection is no longer needed. We use two lower-level MCP SDK elements for this: `sse_client` and `ClientSession`:

```python
from mcp.client.sse import sse_client
from mcp import ClientSession

async def connect_all(self):
    for name, sse_url in self.endpoints.items():
        read, write = await self._exit_stack.enter_async_context(sse_client(sse_url))
        session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self.sessions[name] = session
        
async def disconnect_all(self):
    await self._exit_stack.aclose()
```

> **A Note on `ClientSession` and `sse_client`**
>
> We're working directly with `ClientSession` and `sse_client` here — rather than using a high-level abstraction like `fastmcp` — for a reason.
>
> - `sse_client()` opens a Server-Sent Events connection and returns a read/write stream pair.
> - `ClientSession` wraps those streams into a structured MCP session, offering methods like `initialize()`, `list_tools()`, and `call_tool()`.
>
> Libraries like `fastmcp` abstract this away, but doing it manually helps demystify how MCP “talks” — and shows just how lightweight and modular the protocol actually is.


Finally, we need a method that produces a list of all tools from all server connections and another method to route a call to a given tool:

```python
async def list_all_tools(self):
    all_tools = {}
    for name, session in self.sessions.items():
        tools_response = await session.list_tools()
        all_tools[name] = tools_response.tools
    return all_tools

async def call(self, server_name, tool_name, args):
    session = self.sessions.get(server_name)
    if not session:
        raise ValueError(f"No session for server '{server_name}'")
    return await session.call_tool(tool_name, args)
```

That's already it for the client. We now move on to the third and final building block, the host.

## Exemplary Host

To complete our MCP architecture, we need a way to query an LLM and let it decide which tool to invoke. In this example, we'll use OpenAI's API, but you can easily swap it for any other LLM service or local model server—as long as the model is powerful enough to produce structured output and undeerstand our basic instructions.

```python
from openai import AsyncOpenAI

# Replace this with your real key
openai_key = "your-api-key-here"

# Method to query the LLM
async def query_llm(prompt):
    llm_client = AsyncOpenAI(api_key=openai_key)
    return await llm_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
```

To keep the host compact and focused, we define a single function that accepts a user query and handles the rest. The function:

- Initializes connections to all servers via our MCP client
- Fetches the available tools from all registered servers
- Constructs a prompt for the LLM including the tools and the user query
- Sends the prompt to the LLM
- Parses the LLM's structured JSON response to determine which tool to call
- Calls the selected tool and returns the result
- Gracefully disconnects from all servers

```python
import json

async def run_host_query(user_input: str):
    endpoints = {
        "math_server": "http://127.0.0.1:58000/sse",
        "weather_server": "http://127.0.0.1:58001/sse"
    }

    mcp_client = MultiServerClient(endpoints)
    print("Connect to servers...")
    await mcp_client.connect_all()

    tools_by_server = await mcp_client.list_all_tools()
    tool_summary = []
    for server, tools in tools_by_server.items():
        for tool in tools:
            tool_summary.append(
                f"server: {server}, tool: {tool.name}, description: {tool.description} input schema: {tool.inputSchema}"
            )
    print("Tool summaries:", tool_summary)

    # Prompt to help LLM choose a tool
    prompt = f"""
You are a tool routing assistant. Choose the best tool for the user query. Available tools:

{chr(10).join(tool_summary)}

Given the user query: \"{user_input}\"
Respond only with a JSON object like:
{{"server": "math_server", "tool": "add", "args": {{"a": 3, "b": 5}}}}
"""

    response = await query_llm(prompt)
    raw_content = response.choices[0].message.content

    # Clean markdown formatting if included
    cleaned_content = raw_content.strip().strip("```json").strip("```").strip()
    parsed = json.loads(cleaned_content)
    print("LLM chose:", parsed)

    result = await mcp_client.call(parsed["server"], parsed["tool"], parsed["args"])
    print("Tool result:", result)

    await mcp_client.disconnect_all()
```

This host logic works well in a Jupyter notebook (useful if you want to adjust settings or try out things) or Python script. It’s intended to be a simple but clear, sequential example of how an LLM-powered host can dynamically interact with modular MCP services based on a single user query.

## Results and Wrap-Up

Once everything is up and running, you can call something like:

```python
await run_host_query("How many hours do 5 days have?")
```

And from the print statements, you'll get output like this:

```
Connect to servers...
Tool summaries: [
  "server: math_server, tool: add, description: Adds two integers input schema: {'type': 'object', 'required': ['a', 'b'], 'properties': {'a': {'type': 'integer', 'description': 'First number'}, 'b': {'type': 'integer', 'description': 'Second number'}}}", 
  "server: math_server, tool: multiply, description: Multiplies two integers input schema: {'type': 'object', 'required': ['a', 'b'], 'properties': {'a': {'type': 'integer', 'description': 'First number'}, 'b': {'type': 'integer', 'description': 'Second number'}}}", 
  "server: weather_server, tool: get_weather, description: Returns fake weather input schema: {'type': 'object', 'required': ['city'], 'properties': {'city': {'type': 'string', 'description': 'City to get the weather for'}}}"
]
LLM chose: {'server': 'math_server', 'tool': 'multiply', 'args': {'a': 5, 'b': 24}}
Tool result: meta=None content=[TextContent(type='text', text='120', annotations=None)] isError=False
```

Here's what we see:
- The servers respond to the `list_tools` request via the client, exactly as expected.
- The LLM, using that tool metadata, selects the appropriate tool for the task.
- The tool is called based on the LLM’s structured output — and gives the correct answer: 5 × 24 = **120** hours.

## What’s Next?

Our setup above is intentionally minimal. It’s meant to *illustrate* the inner workings of MCP using multiple servers and low-level SDK constructs. That clarity comes at the cost of abstraction — which is where higher-level libraries like `FastMCP` typically shine.

Still, there’s plenty of room to build on what we’ve done:  
- Automatically register tools via introspection, as hinted in the server section — to reduce redundancy and streamline tool definition.
- Replace SSE with Streamable HTTP transport, the modern alternative now supported in MCP. (We may explore this in a follow-up article.)
- Swap in your own LLM, whether it's a local model or a hosted one — or even build a fully self-hosted chain-of-reasoning setup.
- Integrate libraries like LangChain or LangGraph to formalize and extend the system — using structured format instructions derived from tool metadata, and building more robust, composable prompting workflows.

We’d love to hear your thoughts — especially if you’re experimenting with MCP in more complex or production-grade setups. Just [Email us](mailto:kontakt@seasparks.de).

Thanks for following along — and stay curious. Our AI journey is just getting started.

