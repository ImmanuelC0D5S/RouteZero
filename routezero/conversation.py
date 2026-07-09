from typing import TypedDict
from uuid import uuid4

from routezero.config import Settings


class PipelineState(TypedDict):
    user_prompt: str
    prompt_embedding: list[float]
    cache_hit: bool
    routing_target: str
    task_type: str
    model_response: str
    verification_passed: bool
    execution_latency_ms: float
    conversation_id: str
    turn_index: int


class ConversationStore:
    def __init__(self, window_size: int = 10) -> None:
        self._sessions: dict[str, list[dict[str, str]]] = {}
        self._window_size = window_size
        self._settings = Settings()

    def get_history(self, conversation_id: str) -> list[dict[str, str]]:
        return self._sessions.get(conversation_id, [])

    def append_turn(
        self, conversation_id: str, role: str, content: str
    ) -> None:
        if conversation_id not in self._sessions:
            self._sessions[conversation_id] = []
        self._sessions[conversation_id].append(
            {"role": role, "content": content}
        )
        if len(self._sessions[conversation_id]) > self._window_size:
            self._sessions[conversation_id] = (
                self._sessions[conversation_id][-self._window_size :]
            )

    def build_contextual_prompt(
        self, conversation_id: str, new_prompt: str
    ) -> str:
        history = self.get_history(conversation_id)
        lines: list[str] = []
        lines.append("[System]")
        lines.append("You are RouteZero, an adaptive routing engine.")
        lines.append("")
        if history:
            lines.append("[Conversation history]")
            for turn in history:
                role_label = "User" if turn["role"] == "user" else "Assistant"
                lines.append(f"{role_label}: {turn['content']}")
            lines.append("")
        lines.append("[Current query]")
        lines.append(f"User: {new_prompt}")
        return "\n".join(lines)

    def clear_session(self, conversation_id: str) -> None:
        """Clear the turn history for a given session."""
        self._sessions.pop(conversation_id, None)

    def new_session(self) -> str:
        conversation_id = uuid4().hex
        self._sessions[conversation_id] = []
        return conversation_id
