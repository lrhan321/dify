from typing import Any
from core.tools.errors import ToolProviderCredentialValidationError
from core.tools.provider.builtin_tool_provider import BuiltinToolProviderController
from core.tools.provider.builtin.qichacha.tools.ecinfo_search import QichachaTool


class QichachaProvider(BuiltinToolProviderController):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        try:

            QichachaTool().fork_tool_runtime(
                runtime={
                    "credentials": credentials,
                }
            ).validate_credentials(
                user_id="",
                parameters={
                    "search_key": "91320594088140947F"
                },
            )
        except Exception as e:
            raise ToolProviderCredentialValidationError(str(e))
