from typing import Any, Union
from httpx import get

from core.tools.entities.tool_entities import ToolInvokeMessage
from core.tools.errors import ToolProviderCredentialValidationError
from core.tools.tool.builtin_tool import BuiltinTool
from core.tools.provider.builtin.qichacha._qichacha_tool_base import build_headers


class QichachaTool(BuiltinTool):
    _base_url = "https://api.qichacha.com/ECIInfoVerify/GetInfo"

    def _invoke(
        self,
        user_id: str,
        tool_parameters: dict[str, Any],
    ) -> Union[ToolInvokeMessage, list[ToolInvokeMessage]]:
        """
        invoke tools
        """
        search_key = tool_parameters.get("search_key", "")
        if not search_key:
            return self.create_text_message("Please input query")

        key = self.runtime.credentials.get("key", "")
        secret_key = self.runtime.credentials.get("secret_key", "")
        if not key or not secret_key:
            raise ToolProviderCredentialValidationError("Please input key and secret_key")
        
        params = {"key": key, "searchKey": search_key}
        headers = build_headers(key, secret_key)
        response = get(self._base_url, params=params, headers=headers, timeout=20)

        result = response.json()
        status_code = result.get("Status", "0")
        message = result.get("Message", "")

        if status_code != "200":
            if "【有效请求】" in message:
                return self.create_text_message(message)
            else:
                raise ToolProviderCredentialValidationError(f"Query Failed: {message}")

        # validate credentials
        if tool_parameters.get("validate"):
            return self.create_text_message("Credentials validation successful.")

        data = result.get("Result", {})
        if not data:
            return self.create_text_message("No result found")

        # convert to string and summarize
        text = self.format_dict(data)

        return self.create_text_message(self.summary(user_id, text))

    def validate_credentials(self, parameters: dict[str, Any], **kwargs) -> None:
        parameters["validate"] = True
        self._invoke(tool_parameters=parameters, **kwargs)

    def format_dict(self, data: dict, indent=0):
        """
        Format the dictionary into a string recursively.

        """
        formatted_str = ""
        indent_space = "    " * indent  # 4 spaces per indentation level

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    formatted_str += f"{indent_space}{key}:\n"
                    formatted_str += self.format_dict(value, indent + 1)
                else:
                    formatted_value = value if value is not None else "null"
                    formatted_str += f"{indent_space}{key}: {formatted_value}\n"
        elif isinstance(data, list):
            for idx, item in enumerate(data, 1):
                if isinstance(item, (dict, list)):
                    formatted_str += f"{indent_space}- Item {idx}:\n"
                    formatted_str += self.format_dict(item, indent + 1)
                else:
                    formatted_value = item if item is not None else "null"
                    formatted_str += f"{indent_space}- {formatted_value}\n"
        else:
            formatted_str += f"{indent_space}{data}\n"

        return formatted_str
