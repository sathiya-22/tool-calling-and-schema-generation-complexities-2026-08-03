import pytest
import asyncio
from tool_registry import ToolRegistry, ToolCall
from pydantic import ValidationError

@pytest.fixture
def registry():
    return ToolRegistry()

def test_register_tool_and_schema_generation(registry):
    @registry.register_tool
    def my_test_tool(arg1: str, arg2: int = 10, arg3: bool = False) -> str:
        """
        A simple test tool.
        Args:
            arg1 (str): The first argument.
            arg2 (int, optional): The second argument. Defaults to 10.
            arg3 (bool): A boolean flag.
        """
        return f"{arg1}-{arg2}-{arg3}"

    schemas = registry.get_tool_schema_json()
    assert len(schemas) == 1
    schema = schemas[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "my_test_tool"
    assert "A simple test tool." in schema["function"]["description"]

    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert "arg1" in params["properties"]
    assert params["properties"]["arg1"]["type"] == "string"
    assert "arg1" in params["required"]
    assert params["properties"]["arg1"]["description"] == "The first argument."

    assert "arg2" in params["properties"]
    assert params["properties"]["arg2"]["type"] == "integer"
    assert params["properties"]["arg2"]["default"] == 10
    assert "arg2" not in params["required"]
    assert params["properties"]["arg2"]["description"] == "The second argument."

    assert "arg3" in params["properties"]
    assert params["properties"]["arg3"]["type"] == "boolean"
    assert "arg3" in params["required"]
    assert params["properties"]["arg3"]["description"] == "A boolean flag."


def test_register_duplicate_tool_raises_error(registry):
    @registry.register_tool
    def duplicate_tool():
        pass
    
    with pytest.raises(ValueError, match="already registered"):
        @registry.register_tool
        def duplicate_tool():
            pass

@pytest.mark.asyncio
async def test_execute_sync_tool_call_success(registry):
    @registry.register_tool
    def add(a: int, b: int) -> int:
        """Adds two numbers."""
        return a + b

    tool_call = ToolCall(tool_name="add", arguments={"a": 5, "b": 3})
    result = await registry.execute_tool_call(tool_call)
    assert result == 8

@pytest.mark.asyncio
async def test_execute_async_tool_call_success(registry):
    @registry.register_tool
    async def multiply(a: int, b: int) -> int:
        """Multiplies two numbers asynchronously."""
        await asyncio.sleep(0.01) # Simulate async work
        return a * b

    tool_call = ToolCall(tool_name="multiply", arguments={"a": 5, "b": 3})
    result = await registry.execute_tool_call(tool_call)
    assert result == 15

@pytest.mark.asyncio
async def test_execute_tool_call_missing_required_arg_raises_error(registry):
    @registry.register_tool
    def subtract(a: int, b: int) -> int:
        """Subtracts two numbers."""
        return a - b

    tool_call = ToolCall(tool_name="subtract", arguments={"a": 10}) # Missing 'b'
    with pytest.raises(ValueError, match="received invalid arguments"):
        await registry.execute_tool_call(tool_call)

@pytest.mark.asyncio
async def test_execute_tool_call_unknown_tool_raises_error(registry):
    tool_call = ToolCall(tool_name="non_existent_tool", arguments={"a": 1})
    with pytest.raises(ValueError, match="not found in registry"):
        await registry.execute_tool_call(tool_call)

@pytest.mark.asyncio
async def test_execute_tool_call_with_default_args(registry):
    @registry.register_tool
    def greet(name: str, salutation: str = "Hello") -> str:
        """Greets a person."""
        return f"{salutation}, {name}!"

    tool_call_default = ToolCall(tool_name="greet", arguments={"name": "Alice"})
    result_default = await registry.execute_tool_call(tool_call_default)
    assert result_default == "Hello, Alice!"

    tool_call_override = ToolCall(tool_name="greet", arguments={"name": "Bob", "salutation": "Hi"})
    result_override = await registry.execute_tool_call(tool_call_override)
    assert result_override == "Hi, Bob!"

@pytest.mark.asyncio
async def test_tool_execution_runtime_error(registry):
    @registry.register_tool
    def divide_by_zero(numerator: int, denominator: int) -> float:
        """Divides two numbers, can cause error."""
        return numerator / denominator

    tool_call = ToolCall(tool_name="divide_by_zero", arguments={"numerator": 10, "denominator": 0})
    with pytest.raises(RuntimeError, match="Error executing tool 'divide_by_zero'"):
        await registry.execute_tool_call(tool_call)

@pytest.mark.asyncio
async def test_execute_tool_call_invalid_argument_type_raises_error(registry):
    @registry.register_tool
    def process_number(value: int) -> int:
        """Processes an integer."""
        return value * 2

    tool_call = ToolCall(tool_name="process_number", arguments={"value": "not_an_int"})
    with pytest.raises(ValueError, match="received invalid arguments"):
        await registry.execute_tool_call(tool_call)

@pytest.mark.asyncio
async def test_execute_tool_call_extra_argument_raises_error(registry):
    @registry.register_tool
    def simple_func(param1: str) -> str:
        """A simple function."""
        return param1

    tool_call = ToolCall(tool_name="simple_func", arguments={"param1": "test", "extra_param": "unexpected"})
    with pytest.raises(ValueError, match="received invalid arguments"):
        await registry.execute_tool_call(tool_call)
