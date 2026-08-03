import asyncio
import json
from tool_registry import ToolRegistry, ToolCall

# Initialize the tool registry
registry = ToolRegistry()

# Define some example tools using the decorator
@registry.register_tool
def get_current_weather(location: str, unit: str = "celsius") -> dict:
    """
    Get the current weather in a given location.

    Args:
        location (str): The city and state, e.g. San Francisco, CA
        unit (str, optional): The unit of temperature. Can be 'celsius' or 'fahrenheit'. Defaults to 'celsius'.
    
    Returns:
        dict: A dictionary containing weather information.
    """
    print(f"--- Executing get_current_weather for {location} in {unit} ---")
    # In a real scenario, this would call an external API
    if location == "London, UK":
        return {"location": location, "temperature": "10", "unit": unit, "forecast": "cloudy"}
    elif location == "New York, USA":
        return {"location": location, "temperature": "68" if unit == "fahrenheit" else "20", "unit": unit, "forecast": "sunny"}
    else:
        return {"location": location, "temperature": "unknown", "unit": unit, "forecast": "N/A"}

@registry.register_tool
async def search_web(query: str) -> str:
    """
    Performs a web search for a given query.

    Args:
        query (str): The search query.
    
    Returns:
        str: A summary of the search results.
    """
    print(f"--- Executing search_web for '{query}' ---")
    await asyncio.sleep(0.5) # Simulate async operation
    if "latest AI news" in query.lower():
        return "Latest AI news: Breakthroughs in large language models and multimodal AI."
    return f"Simulated search results for: '{query}'"

@registry.register_tool
def send_email(recipient: str, subject: str, body: str):
    """
    Sends an email to a specified recipient.

    Args:
        recipient (str): The email address of the recipient.
        subject (str): The subject line of the email.
        body (str): The main content of the email.
    """
    print(f"--- Executing send_email to {recipient} ---")
    print(f"Subject: {subject}")
    print(f"Body: {body[:50]}...")
    return f"Email sent successfully to {recipient}."


async def main():
    print("--- Generating Tool Schemas ---")
    schemas_json = registry.get_tool_schema_json()
    print(json.dumps(schemas_json, indent=2))

    print("\n--- Simulating LLM Tool Calls (using fixture data) ---")

    # Fixture 1: Valid tool call
    fixture_tool_call_1 = ToolCall(
        tool_name="get_current_weather",
        arguments={"location": "London, UK", "unit": "celsius"}
    )
    print(f"\nSimulated LLM call: {fixture_tool_call_1.model_dump_json(indent=2)}")
    try:
        result = await registry.execute_tool_call(fixture_tool_call_1)
        print(f"Tool execution result: {result}")
    except Exception as e:
        print(f"Error during tool execution: {e}")

    # Fixture 2: Another valid tool call (async)
    fixture_tool_call_2 = ToolCall(
        tool_name="search_web",
        arguments={"query": "latest AI news"}
    )
    print(f"\nSimulated LLM call: {fixture_tool_call_2.model_dump_json(indent=2)}")
    try:
        result = await registry.execute_tool_call(fixture_tool_call_2)
        print(f"Tool execution result: {result}")
    except Exception as e:
        print(f"Error during tool execution: {e}")

    # Fixture 3: Tool call with default argument
    fixture_tool_call_3 = ToolCall(
        tool_name="get_current_weather",
        arguments={"location": "New York, USA"} # unit defaults to 'celsius'
    )
    print(f"\nSimulated LLM call: {fixture_tool_call_3.model_dump_json(indent=2)}")
    try:
        result = await registry.execute_tool_call(fixture_tool_call_3)
        print(f"Tool execution result: {result}")
    except Exception as e:
        print(f"Error during tool execution: {e}")

    # Fixture 4: Tool call with missing required argument (will raise error)
    fixture_tool_call_4 = ToolCall(
        tool_name="get_current_weather",
        arguments={"unit": "fahrenheit"} # Missing 'location'
    )
    print(f"\nSimulated LLM call: {fixture_tool_call_4.model_dump_json(indent=2)}")
    try:
        result = await registry.execute_tool_call(fixture_tool_call_4)
        print(f"Tool execution result: {result}")
    except Exception as e:
        print(f"Error during tool execution: {e}")
        # Expected error: TypeError due to missing required argument

    # Fixture 5: Tool call to a non-existent tool
    fixture_tool_call_5 = ToolCall(
        tool_name="unregistered_tool",
        arguments={"param": "value"}
    )
    print(f"\nSimulated LLM call: {fixture_tool_call_5.model_dump_json(indent=2)}")
    try:
        result = await registry.execute_tool_call(fixture_tool_call_5)
        print(f"Tool execution result: {result}")
    except Exception as e:
        print(f"Error during tool execution: {e}")
        # Expected error: ValueError: Tool 'unregistered_tool' not found

    # Fixture 6: Another valid tool call
    fixture_tool_call_6 = ToolCall(
        tool_name="send_email",
        arguments={
            "recipient": "agent@example.com",
            "subject": "Important Update",
            "body": "This is a test email sent from the agentic system. Please disregard if you are not the intended recipient."
        }
    )
    print(f"\nSimulated LLM call: {fixture_tool_call_6.model_dump_json(indent=2)}")
    try:
        result = await registry.execute_tool_call(fixture_tool_call_6)
        print(f"Tool execution result: {result}")
    except Exception as e:
        print(f"Error during tool execution: {e}")


if __name__ == "__main__":
    asyncio.run(main())
