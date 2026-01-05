"""
Mathematical toolkit for symbolic computations using SymPy.
Includes methods for algebraic manipulation, calculus, and linear algebra.
"""

import json
from collections.abc import Callable

import sympy as sp

from utu.config import ToolkitConfig
from utu.utils import get_logger
from utu.tools import AsyncBaseToolkit, register_tool

logger = get_logger(__name__)


class MathToolkit(AsyncBaseToolkit):
    def __init__(self, config: ToolkitConfig = None, default_variable: str = 'x') -> None:
        super().__init__(config)
        self.default_variable = default_variable
        logger.info(f"Default variable set to: {self.default_variable}")

    def _handle_exception(self, method_name: str, e: Exception, context: dict = None) -> str:
        """Handle exceptions and return formatted error message.
        
        Args:
            method_name: Name of the method where the error occurred
            e: The exception that was raised
            context: Optional dictionary with additional context (e.g., expression, parsed_expr)
        """
        error_msg = f"Error in {method_name}: {str(e)}"
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            error_msg = f"{error_msg} (context: {context_str})"
        logger.error(error_msg)
        return json.dumps({"error": error_msg}, ensure_ascii=False)

    @register_tool
    async def calculator(self, expression: str) -> str:
        r"""Evaluates a mathematical expression.

        Args:
            expression (str): The mathematical expression to evaluate,
                provided as a string.

        Returns:
            str: JSON string containing the result of the evaluation in the
                `"result"` field. If an error occurs, the JSON string will
                include an `"error"` field with the corresponding error
                message.
        """
        try:
            expr = sp.parsing.sympy_parser.parse_expr(expression, evaluate=True)
            
            # Check if expression contains free symbols (variables)
            free_symbols = expr.free_symbols
            if free_symbols:
                symbols_str = ", ".join(str(s) for s in free_symbols)
                context = {
                    "expression": expression,
                    "parsed_expr": str(expr),
                    "free_symbols": symbols_str
                }
                error_msg = (
                    f"Cannot evaluate expression with undefined variables: {symbols_str}. "
                    f"Expression: '{expression}' -> '{expr}'. "
                    f"Please provide numeric values for all variables."
                )
                logger.error(f"Error in calculator: {error_msg}")
                return json.dumps({"error": error_msg}, ensure_ascii=False)
            
            num = sp.N(expr, 15)  # enough precision
            val = float(num)
            formatted = ("{:.6f}".format(val)).rstrip('0').rstrip('.')
            return json.dumps({"result": formatted}, ensure_ascii=False)
        except (ValueError, TypeError) as e:
            # Handle conversion errors with more context
            context = {
                "expression": expression,
                "parsed_expr": str(expr) if 'expr' in locals() else "failed to parse",
                "error_type": type(e).__name__
            }
            error_msg = (
                f"Cannot convert expression to float. "
                f"Expression: '{expression}' -> '{context['parsed_expr']}'. "
                f"Error: {str(e)}"
            )
            logger.error(f"Error in calculator: {error_msg}")
            return json.dumps({"error": error_msg}, ensure_ascii=False)
        except Exception as e:
            # Handle other exceptions with context
            context = {
                "expression": expression,
                "parsed_expr": str(expr) if 'expr' in locals() else "failed to parse",
                "error_type": type(e).__name__
            }
            return self._handle_exception("calculator", e, context)