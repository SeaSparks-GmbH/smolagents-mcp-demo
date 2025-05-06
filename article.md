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

The server can be started as usual just with `python library_server.py`.

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

- A full list of available tools (in this case: `get_library` and `get_document`)
- Each tool’s name, description, input schema, and output type
- No hardcoded API specs — just introspectable tools, as described by the server itself.

**Why This Matters**  
This example may look minimal, but it’s doing a lot:

- Server discovery
- Schema understanding
- Getting `tools` ready to be inspected and used by a SmolAgent

So it is simple but powerful. Whether you’re building a UI, wiring an agent, or testing toolchains, this client handles the connection and lets you focus on what to do with the tools — which is exactly what we'll tackle next.

## The Host: A Super-Minimal Smolagent Setup

Now it’s time to bring the pieces together.

To keep things focused and beginner-friendly, we’ll set up just a single agent, using Smolagents’ adorable `CodeAgent`. This agent can inspect tools from the MCP server, generate Python code to use those tools, read and reason over results, and iterate until it finds a valid solution.

Yes, it's only *one* agent — but it's already doing a lot.

**Why CodeAgent?**  
Because most powerful LLMs are very good at writing and fixing Python code. With CodeAgent, the model is allowed to generate runnable Python, observe its output, handle errors, and retry as needed — until the task is done.

Even simple tasks can involve a lot of interaction and self-correction, which leads us to an important note…

**Watch your tokens**  
Agentic calls are more expensive than standard one-shot LLM completions. Even in the absence of long texts, a single agent cycle might:
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
- Runs the agent on a given `user_input`.

This is about as minimal as it gets—but it already supports MCP tool use, Python execution, retry loops, and full reasoning.

In the final section, we’ll show some fun examples—real queries, real agent behavior, real results.

Stay tuned. It’s going to be worth it.

## Results: Conversations with a Quantum Squirrel (and Friends)

With everything wired up: our DocumentServer, the MCPClient, and the minimal "Host" aka CodeAgent, it’s finally time to see the system in action.

We give the agent access to a small selection of fantastical papers from our research library, including three ground-breaking works on turnips, quantum squirrels and dark coffee energy, respectively.

As the library is (or rather was, until the GitHub repo became public) private, we are certain that at the time of writing, neither gpt-4o-mini nor any other LLM powering the agent has seen the contents. Hence to provide substantial answer to the queries, the agent **must** get a hand on the actual documents from this library. 
So let’s see how and what it made of them.

**Discovering the Library**

With 
```python
run_host_query("Which documents are available?")
```
we get the agent working on the given task. Note that in our request, we didn't use words like "library" or "title" (as in document title) and, foremeost, didn't advise the agent to use any of the available tools. Actually, on the side of the user, any knowlegde about tools is completely unnecessary - it is sufficient to know that there is a library somewhere.

(Agent output skipped for brevity—but it did involve connecting, printing tool names, drafting Python code, fixing an import typo, trying again...)
Eventually, the agent presented:

```
['The Turnip That Spoke Latin.pdf',
 'Quantum Squirrel Report.pdf',
 'A Note on the Conservation of Coffee Energy.pdf']
```

So, the agentic, tool-driven discovery via MCP worked. And seriously  — what a promising list!

**The Quantum Squirrel Report**  
Now that we know the titles available in the library, we are free to ask the agent basically anything about them. Maybe you are intrigued to learn more about the quantum squirrel and execute:
```python
run_host_query("Give me a summary of the quantum squrrel report, please.")
```
Uups, we apperently lost an "i" - may it serve a deeper purpose in another parallel universe. Our agent, using the `get_docoment` tool, however, is unimpressed by this ommission and gives the result:
Again, we skip the full agent output (but in the article: include a trimmed version! show some retries, maybe a small stack trace + recovery—Smolagents are charming when they mess up a bit and recover on their own).  

```
"The Quantum Squirrel Report is a book report by Timothy J. Baxter.  
The report discusses a book about a squirrel named Quince who lives 
behind a school and possesses unusual abilities, such as existing in 
a probability cloud. He interacts with a boy named Nate, talking about
complex themes like entanglement and reality. The story conveys bizarre
events including a vision of the future where squirrels have evolved.  
Overall, the report highlights the confusion and intrigue of the narrative
while learning about quantum physics.  
Timothy rates the book four out of five acorns."
```

Sounds promisingly hilariously good. So we immediately execute
```python
run_host_query("I would like to see the full text about the quantum squirrel, please.")
```
and get
```
My Book Report on The Quantum Squirrel
by Timothy J. Baxter, Grade 7
My teacher said we had to write a report on a book, so I picked one called
The Quantum Squirrel, which I found on my cousin’s old Kindle. I thought
it was about science and squirrels, which are both cool. The book starts off
kinda weird but I liked it. It had chapters and also footnotes sometimes [...]
```
(and no, you won't get to see the full text here - just get a clone of the GitHub repo and enjoy).

**A Note on Agent Behavior**  
One of the joys of working with CodeAgent is watching how it:

- Chooses which tool to use
- Writes Python code to call it
- Parses the output
- Then uses that result to generate natural-language responses 

And when things go wrong—missing imports, bad arguments, or decoding errors—it notices, fixes, and tries again. All in real-time.

These aren’t just static LLM calls. This is active, iterative reasoning—with a dash of personality. 🤗.

## Closing Thoughts
You’ve now seen how a low-level MCP server, a simple CodeAgent host, and a couple of whimsical PDFs can produce a miniature agent system with real capabilities and charm.

If you want to run it yourself, the full code is available on GitHub (link below).

In the meantime—go write your own talking vegetable memoirs or particle-riding rodent tales. Your agents are ready to read them.

## What’s Next?

Apart from the stories, our setup above is intentionally minimal. It’s meant to *illustrate* the workings of MCP together with a SmolAgent.

So there’s plenty of room to build on what we’ve done:  
- Automatically register tools via introspection, as hinted in the server section — to reduce redundancy and streamline tool definition.
- Replace SSE with Streamable HTTP transport, the modern alternative now supported in MCP. (We may explore this in a follow-up article.)
- Swap in your own LLM, whether it's a local model or a hosted one — or even build a fully self-hosted chain-of-reasoning setup.
- Integrate libraries like LangChain or LangGraph to formalize and extend the system — using structured format instructions derived from tool metadata, and building more robust, composable prompting workflows.

We’d love to hear your thoughts — especially if you’re experimenting with agents and MCP in more complex or production-grade setups. Just [Email us](mailto:kontakt@seasparks.de).

Thanks for following along — and stay curious. Our AI journey is just getting started.