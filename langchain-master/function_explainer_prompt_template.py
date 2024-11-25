from langchain.prompts import StringPromptTemplate
from pydantic import BaseModel, field_validator
import inspect

def get_source_code(function_name):
    return inspect.getsource(function_name)

def test_add():
    return 1 + 1

PROMPT = """\
Given the function name and source code, generate an English language explanation of the function.
Function Name: {function_name}
Source Code:
{source_code}
Explanation:
"""

class FunctionExplainerPromptTemplate(StringPromptTemplate, BaseModel):
    """A custom prompt template that takes in the function name as input, and formats the prompt template to provide the source code of the function"""

    @field_validator("input_variables")
    def validate_input_variables(cls, v):
        """校验输入的变量是否正确"""
        if len(v) != 1 or "function_name" not in v:
            raise ValueError("function_name must be the only input_variable.")
        return v

    def format(self, **kwargs) -> str:
        source_code = get_source_code(kwargs["function_name"])
        prompt = PROMPT.format(
            function_name = kwargs["function_name"].__name__, source_code=source_code
        )
        return prompt

    def _prompt_type(self) -> str:
        return "function-explainer"