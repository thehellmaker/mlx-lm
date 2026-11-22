# Copyright © 2026 Apple Inc.

import json

tool_call_start = "<|content_invoke_tool_json|>"

tool_call_end = "<|end_message|>"


def parse_tool_call(text, tools=None):
    call = json.loads(text.strip())
    if not isinstance(call, dict) or not isinstance(call.get("name"), str):
        raise ValueError(f"Inkling tool call needs a name: {text!r}")
    arguments = call.get("args", call.get("arguments", {}))
    if not isinstance(arguments, dict):
        raise ValueError("Inkling tool arguments must be a JSON object")
    return {"name": call["name"], "arguments": arguments}
