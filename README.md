## Tool Schema Generator & Universal Tool Caller

### The Real Problem
The agentic AI community struggles with the complexities of generating accurate tool schemas from Python functions and ensuring consistent tool calling behavior across diverse LLM providers and frameworks. Developers face issues with:
1.  **Incorrect Schema Generation**: Manually crafting or semi-automatically deriving tool schemas often leads to errors or missing details, especially for complex function signatures with various argument types, defaults, and docstrings.
2.  **Lack of Dynamic Tool Management**: Adding, updating, or removing tools at runtime is cumbersome, often requiring significant code changes or re-initialization.
3.  **Inconsistent Tool Execution**: Different LLM providers (e.g., OpenAI, Gemini, Anthropic) have varying expectations for tool call formats, leading to brittle agent systems that break when switching providers or integrating multiple.

This prototype addresses these pain points by providing a robust, provider-agnostic framework for Python function-to-tool-schema generation and a unified mechanism for executing tool calls, abstracting away LLM-specific nuances.

### Why This Project Shape/Stack Was Chosen
A Python package is the most natural fit for this problem because:
*   **Python is the dominant language** in the agentic AI ecosystem, making a Python-based solution directly usable by the target audience.
*   **Function introspection** in Python is powerful and allows for accurate schema generation.
*   **Package distribution** simplifies integration into existing agent frameworks and projects.
*   **Flexibility**: It allows for both a library approach (integrating into an agent framework) and a standalone utility.

### Setup and Usage Instructions (Zero API Keys Required)

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the example:**
    The `example.py` demonstrates how to define tools, generate schemas, and simulate tool calls using fixture data.

    ```bash
    python example.py
    ```

    You will see output demonstrating:
    *   Generated JSON schemas for sample Python functions.
    *   Simulation of an LLM's tool call request (using a fixture).
    *   Execution of the simulated tool call, including error handling.

### Optional Real-LLM Adapter
This prototype's core logic is LLM-provider agnostic. It focuses on the schema generation and tool execution layers. An optional real-LLM adapter is *not* provided as part of this initial prototype because the problem statement explicitly focuses on the complexities *around* tool calling (schema generation, dynamic management, consistent execution), not on the LLM's role in *generating* the tool call itself. The `example.py` uses fixture data to simulate the LLM's output, allowing the core logic to be demonstrated and tested without any API keys.

If an LLM integration were to be added, it would be a separate component that takes the generated schemas and feeds them to an LLM provider's SDK, then parses the LLM's response into the `ToolCall` format used by this prototype. This separation ensures the core utility remains robust and testable independently.
