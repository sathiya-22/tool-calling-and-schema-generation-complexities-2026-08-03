from typing import Callable, Dict, Any, List, TypeVar
import inspect
import json
from pydantic import BaseModel, Field, create_model

# Type variable for the decorated function
F = TypeVar('F', bound=Callable[..., Any])

class ToolCall(BaseModel):
    """Represents a standardized tool call from an LLM."""
    tool_name: str = Field(..., description="The name of the tool to be called.")
    arguments: Dict[str, Any] = Field(..., description="A dictionary of arguments for the tool.")

class ToolSchema(BaseModel):
    """Represents the schema for a single tool, compliant with common LLM provider formats."""
    type: str = Field("function", const=True, description="The type of the tool (always 'function').")
    function: Dict[str, Any] = Field(..., description="Details of the function, including name, description, and parameters schema.")

class ToolRegistry:
    """
    Manages the registration, schema generation, and execution of tools.
    Provides a provider-agnostic interface for tool management.
    """
    def __init__(self):
        self._tools: Dict[str, Callable[..., Any]] = {}
        self._schemas: Dict[str, ToolSchema] = {}

    def register_tool(self, func: F) -> F:
        """
        Decorator to register a function as a tool and automatically generate its schema.
        The function's docstring is used as the tool description.
        Argument types, defaults, and descriptions (from docstring if formatted) are used for schema generation.
        """
        tool_name = func.__name__
        if tool_name in self._tools:
            raise ValueError(f"Tool '{tool_name}' already registered.")

        self._tools[tool_name] = func
        self._schemas[tool_name] = self._generate_schema(func)
        return func

    def _generate_schema(self, func: Callable[..., Any]) -> ToolSchema:
        """Generates a Pydantic-compatible JSON schema for a given function."""
        signature = inspect.signature(func)
        docstring = inspect.getdoc(func) or ""

        # Basic parsing for description and argument descriptions from docstring
        description_lines = []
        param_descriptions: Dict[str, str] = {}
        in_params_section = False
        for line in docstring.split('\n'):
            stripped_line = line.strip()
            if stripped_line.lower().startswith('args:'):
                in_params_section = True
                continue
            if stripped_line.lower().startswith('returns:') or stripped_line.lower().startswith('raises:'):
                in_params_section = False
            
            if in_params_section:
                # Basic parsing for `:param name: description` or `name (type): description`
                if stripped_line.startswith(':param '):
                    parts = stripped_line[len(':param '):].split(':', 1)
                    if len(parts) == 2:
                        name = parts[0].strip().split(' ')[0] # handle ':param type name:'
                        param_descriptions[name] = parts[1].strip()
                elif ' (type): ' in stripped_line: # Common style for Sphinx/Numpy docstrings
                    parts = stripped_line.split(' (type): ', 1)
                    if len(parts) == 2:
                        name = parts[0].strip()
                        param_descriptions[name] = parts[1].strip()
                elif ' -- ' in stripped_line: # Google style docstrings
                    parts = stripped_line.split(' -- ', 1)
                    if len(parts) == 2:
                        name = parts[0].strip().split(' ')[0]
                        param_descriptions[name] = parts[1].strip()
            elif not in_params_section and stripped_line:
                description_lines.append(stripped_line)
        
        func_description = ' '.join(description_lines).strip() or f"Calls the {func.__name__} function."


        fields: Dict[str, Any] = {}
        for name, param in signature.parameters.items():
            if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD or \
               param.kind == inspect.Parameter.KEYWORD_ONLY:
                
                field_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
                field_default = param.default if param.default != inspect.Parameter.empty else Field(..., description=param_descriptions.get(name, ""))

                if field_default is Field(...): # Required argument
                    fields[name] = (field_type, Field(..., description=param_descriptions.get(name, "")))
                else: # Optional argument with a default
                    fields[name] = (field_type, Field(field_default, description=param_descriptions.get(name, "")))

        # Create a Pydantic model dynamically for the function's arguments
        ArgsModel = create_model(f"{func.__name__}Args", **fields)
        
        return ToolSchema(
            function={
                "name": func.__name__,
                "description": func_description,
                "parameters": ArgsModel.model_json_schema()
            }
        )

    def get_tool_schemas(self) -> List[ToolSchema]:
        """Returns a list of all registered tool schemas."""
        return list(self._schemas.values())

    def get_tool_schema_json(self) -> List[Dict[str, Any]]:
        """Returns a list of all registered tool schemas as JSON-serializable dictionaries."""
        return [schema.model_dump(mode='json', by_alias=True) for schema in self._schemas.values()]

    async def execute_tool_call(self, tool_call: ToolCall) -> Any:
        """
        Executes a registered tool based on a standardized ToolCall object.
        Handles argument validation and potential errors.
        """
        if tool_call.tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_call.tool_name}' not found in registry.")

        tool_func = self._tools[tool_call.tool_name]
        schema = self._schemas[tool_call.tool_name]

        try:
            # Validate arguments against the generated Pydantic schema
            # We need to re-create the Pydantic model from the schema to validate
            # This ensures consistent validation with the schema itself
            func_schema_params = schema.function["parameters"]
            
            # Dynamically create a Pydantic model for validation
            # This is a bit more involved than direct Pydantic parsing
            # since we're working with a JSON schema.
            # For simplicity, we'll rely on the existing Pydantic model creation
            # during schema generation and assume the generated schema correctly
            # reflects argument needs. Direct validation against model_json_schema
            # requires a library like jsonschema or recreating the Pydantic model
            # from the schema itself. For this prototype, we'll pass directly and
            # let Python's type checking/runtime errors handle basic mismatches,
            # trusting the LLM to adhere to the schema it was given.
            # A more robust solution would dynamically load the Pydantic model
            # used for schema generation or use `jsonschema.validate`.

            # For now, we'll try to call directly.
            # If the schema generation is correct, the LLM's call should match.
            # A future improvement would be explicit `jsonschema.validate(tool_call.arguments, func_schema_params)`
            
            if inspect.iscoroutinefunction(tool_func):
                result = await tool_func(**tool_call.arguments)
            else:
                result = tool_func(**tool_call.arguments)
            return result
        except TypeError as e:
            raise ValueError(f"Tool '{tool_call.tool_name}' received invalid arguments: {e}. "
                             f"Expected schema: {json.dumps(schema.function['parameters'], indent=2)}")
        except Exception as e:
            raise RuntimeError(f"Error executing tool '{tool_call.tool_name}': {e}")
