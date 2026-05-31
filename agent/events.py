from enum import Enum
from dataclasses import dataclass, field
from typing import Any


class AgentEventType(str, Enum):
     # Agent Lifecycle
     AGENT_START = "agent_start"
     AGENT_END = "agent_end"
     AGENT_ERROR = "agent_error"
     
     # Text streaming
     TEXT_DELTA = "text_delta"
     TEXT_COMPLETE = "text_complete"
     
     # LLM interaction
     LLM_REQUEST = "llm_request"
     LLM_RESPONSE = "llm_response"
     LLM_ERROR = "llm_error"


@dataclass
class AgentEvent:
    type: AgentEventType
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def agent_start(cls, message: str) -> AgentEvent:
        return cls(
            type=AgentEventType.AGENT_START,
            data={"message": message},
        )

    @classmethod
    def agent_end(cls, result: Any = None, details: dict[str, Any] | None = None) -> AgentEvent:
        return cls(
            type=AgentEventType.AGENT_END,
            data={"result": result, "details": details or {}},
        )

    @classmethod
    def agent_error(
        cls,
        error: str,
        details: dict[str, Any] | None = None,
    ) -> AgentEvent:
        return cls(
            type=AgentEventType.AGENT_ERROR,
            data={"error": error, "details": details or {}},
        )

    @classmethod
    def text_delta(cls, content: str) -> AgentEvent:
        return cls(
            type=AgentEventType.TEXT_DELTA,
            data={"content": content},
        )

    @classmethod
    def text_complete(cls, content: str) -> AgentEvent:
        return cls(
            type=AgentEventType.TEXT_COMPLETE,
            data={"content": content},
        )

    @classmethod
    def llm_request(cls, messages: list[dict[str, Any]], **kwargs: Any) -> AgentEvent:
        return cls(
            type=AgentEventType.LLM_REQUEST,
            data={"messages": messages, "options": kwargs},
        )

    @classmethod
    def llm_response(cls, response: Any, usage: Any = None) -> AgentEvent:
        return cls(
            type=AgentEventType.LLM_RESPONSE,
            data={"response": response, "usage": usage},
        )

    @classmethod
    def llm_error(cls, error: str, details: dict[str, Any] | None = None) -> AgentEvent:
        return cls(
            type=AgentEventType.LLM_ERROR,
            data={"error": error, "details": details or {}},
        )