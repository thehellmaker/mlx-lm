# Copyright © 2026 Apple Inc.

"""Response format of Inkling 2 models.

Each message the model emits is wrapped in:

    <|message_model|><|content_thinking|>...reasoning...<|end_message|>
    <|message_model|><|content_text|>answer<|end_message|>
    <|message_model|><|content_invoke_tool_json|>{"name":...,"args":{...}}<|end_message|>

The wrappers are dropped; thinking goes to ``reasoning``, text to ``content``
and tool JSON to a tool call. The tool-call exit edge comes from the tool
parser's ``tool_call_end``.
"""

RESPONSE_FORMAT = {
    "initial_state": "normal",
    "transitions": {
        "normal": [
            ("<|message_model|>", "normal"),
            ("<|content_text|>", "normal"),
            ("<|content_thinking|>", "reasoning"),
            ("<|content_invoke_tool_json|>", "tool"),
            ("<|end_message|>", "normal"),
        ],
        "reasoning": [
            ("<|message_model|>", "reasoning"),
            ("<|content_thinking|>", "reasoning"),
            ("<|end_message|>", "normal"),
        ],
    },
}
