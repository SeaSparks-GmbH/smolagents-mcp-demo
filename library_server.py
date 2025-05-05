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

server = Server("DocumentServer")
library_path = "./library"

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

# NOT decorated
def get_library() -> str:
    """Gets a list of documents in the library.
       The output is of the form {"list_of_files":[file1, file2, ...]}

    Args:

    """

    pdf_files = [f for f in os.listdir(library_path) if f.lower().endswith(".pdf")]

    return json.dumps({"list_of_files": pdf_files})

# NOT decorated
def get_document(title: str) -> str:
    """Gets the content of a document.

    Args:
         title: The title of the document, excluding the extension / document type. Call with title="..." as argument.
    """

    pdf_files = json.loads(get_library())['list_of_files']
    pdf_files_cleaned = [f.lower().strip().replace("_", " ") for f in pdf_files]

    clean_title = title.lower().strip().replace("_", " ")

    # Best fuzzy match using token_set_ratio
    result = process.extractOne(
        clean_title,
        pdf_files_cleaned,
        scorer=fuzz.token_set_ratio,
        score_cutoff=50
    )

    if result is None:
        return f"No matching document found for: {title}"

    best_match, score, _ = result

    # Recover original filename
    matched_index = pdf_files_cleaned.index(best_match)
    file_path = os.path.join(library_path, pdf_files[matched_index])

    try:
        with fitz.open(file_path) as doc:
            text = "\n".join(page.get_text() for page in doc)
        return text.strip() or f"Couldn't extract the content of document '{title}'."
    except Exception as e:
        return f"Error reading document '{title}': {e}"


@server.list_tools()
async def list_tools():
    return [
        build_tool_schema(get_library),
        build_tool_schema(get_document),
    ]

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
