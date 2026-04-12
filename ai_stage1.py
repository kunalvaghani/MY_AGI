from __future__ import annotations

"""
Stage 1 local AI assistant.

This file contains the full runtime logic for a small command-line chat app that:
- reads user input,
- sends it to a local Ollama model,
- keeps conversation history in memory for the current process only,
- supports /help, /reset, and /exit,
- exits cleanly on Ctrl+C.

The code is intentionally simple, heavily commented, and self-contained.
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

try:
    # Official Ollama Python client.
    from ollama import Client, ResponseError
except ModuleNotFoundError as import_error:
    Client = None  # type: ignore[assignment]

    class ResponseError(Exception):
        """Fallback definition used only when the ollama package is missing."""

    OLLAMA_IMPORT_ERROR: ModuleNotFoundError | None = import_error
else:
    OLLAMA_IMPORT_ERROR = None


APP_NAME = "Local AI Assistant - Stage 1"
DEFAULT_MODEL = "gemma3:latest"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.2

HELP_TEXT = """Available commands:
  /help   Show this help message
  /reset  Clear the current in-memory conversation
  /exit   Quit the application"""


@dataclass(frozen=True)
class AppConfig:
    """Resolved runtime configuration for the application."""

    model: str
    ollama_host: str = DEFAULT_OLLAMA_HOST
    temperature: float = DEFAULT_TEMPERATURE


class ConfigResolver:
    """Resolve configuration from CLI arguments and environment variables."""

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        """Create the small argument parser used by the app."""
        parser = argparse.ArgumentParser(
            description="Run the Stage 1 local AI assistant against a local Ollama model."
        )
        parser.add_argument(
            "--model",
            dest="model",
            help="Override the Ollama model name for this run.",
        )
        return parser

    @classmethod
    def resolve(
        cls,
        argv: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> AppConfig:
        """
        Resolve configuration in the required priority order.

        Priority:
        1. --model command-line argument
        2. OLLAMA_MODEL environment variable
        3. default model
        """
        parser = cls.build_parser()
        args = parser.parse_args(list(argv) if argv is not None else None)

        env_map = env if env is not None else os.environ

        cli_model = (args.model or "").strip()
        env_model = env_map.get("OLLAMA_MODEL", "").strip()

        if cli_model:
            selected_model = cli_model
        elif env_model:
            selected_model = env_model
        else:
            selected_model = DEFAULT_MODEL

        return AppConfig(model=selected_model)


class OllamaServiceError(Exception):
    """Raised when the app cannot talk to Ollama successfully."""


class OllamaChatClient:
    """Small wrapper around the official Ollama Python client."""

    def __init__(self, host: str) -> None:
        """Create the client or raise a clear error if the package is missing."""
        self.host = host

        if OLLAMA_IMPORT_ERROR is not None or Client is None:
            raise OllamaServiceError(
                "The 'ollama' Python package is not installed. "
                "Run: py -3.11 -m pip install -r requirements.txt"
            )

        # The official client talks to the local Ollama service at this host.
        self._client = Client(host=host)

    def is_reachable(self) -> tuple[bool, str | None]:
        """
        Check whether the Ollama service can be reached.

        We use a lightweight API call. If it fails, return a friendly error string
        instead of letting a raw exception bubble up to the terminal.
        """
        try:
            self._client.list()
            return True, None
        except Exception as exc:  # pragma: no cover - exact client exception varies
            return False, self._format_connection_error(exc)

    def send_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
    ) -> str:
        """
        Send the full in-memory conversation to Ollama and return assistant text.

        The caller passes all messages for the current session so memory remains in
        Python only for this runtime.
        """
        try:
            response = self._client.chat(
                model=model,
                messages=messages,
                stream=False,
                options={"temperature": temperature},
            )
            text = self._extract_text(response)
            if not text:
                raise OllamaServiceError("Ollama returned an empty response.")
            return text
        except OllamaServiceError:
            raise
        except ResponseError as exc:
            status_code = getattr(exc, "status_code", None)
            detail = getattr(exc, "error", str(exc))

            if status_code == 404:
                raise OllamaServiceError(
                    f"Model '{model}' was not found in Ollama. "
                    f"Pull it first with: ollama pull {model}"
                ) from exc

            raise OllamaServiceError(f"Ollama returned an error: {detail}") from exc
        except Exception as exc:  # pragma: no cover - exact client exception varies
            raise OllamaServiceError(self._format_connection_error(exc)) from exc

    def _extract_text(self, response: Any) -> str:
        """Extract assistant text from either object-style or dict-style responses."""
        # Newer versions of the official library expose response.message.content.
        message_obj = getattr(response, "message", None)
        if message_obj is not None:
            content = getattr(message_obj, "content", "")
            if isinstance(content, str):
                return content.strip()

        # Keep a dict fallback in case the library returns mapping-like objects.
        if isinstance(response, dict):
            message = response.get("message", {})
            if isinstance(message, dict):
                content = message.get("content", "")
                if isinstance(content, str):
                    return content.strip()

        raise OllamaServiceError("Could not read the text content from Ollama's response.")

    def _format_connection_error(self, exc: Exception) -> str:
        """Turn a low-level exception into a clear, user-facing message."""
        return (
            f"Could not connect to Ollama at {self.host}. "
            f"Make sure Ollama is installed and running. Details: {exc}"
        )


@dataclass
class SessionMemory:
    """In-memory conversation state for the current process only."""

    messages: list[dict[str, str]] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        """Append a user message to the session history."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Append an assistant message to the session history."""
        self.messages.append({"role": "assistant", "content": content})

    def reset(self) -> None:
        """Clear all in-memory chat history for this run."""
        self.messages.clear()

    def remove_last_message(self) -> None:
        """Remove the newest message, if one exists."""
        if self.messages:
            self.messages.pop()

    def get_messages(self) -> list[dict[str, str]]:
        """Return a safe copy of the stored messages."""
        return [message.copy() for message in self.messages]


class CommandParser:
    """Parse the small set of slash commands for this stage."""

    def parse(self, raw_text: str) -> str:
        """Return one of: help, reset, exit, unknown, message, empty."""
        text = raw_text.strip()

        if not text:
            return "empty"

        if not text.startswith("/"):
            return "message"

        normalized = text.lower()
        if normalized == "/help":
            return "help"
        if normalized == "/reset":
            return "reset"
        if normalized == "/exit":
            return "exit"
        return "unknown"


class Stage1CLIApp:
    """Own the runtime memory, command handling, and command-line chat loop."""

    def __init__(
        self,
        config: AppConfig,
        ollama_client: OllamaChatClient,
        memory: SessionMemory | None = None,
        command_parser: CommandParser | None = None,
    ) -> None:
        """Store the app dependencies in one place."""
        self.config = config
        self.ollama_client = ollama_client
        self.memory = memory if memory is not None else SessionMemory()
        self.command_parser = command_parser if command_parser is not None else CommandParser()

    def process_user_text(self, raw_text: str) -> tuple[bool, str | None]:
        """
        Process a single line of terminal input.

        Returns:
            (keep_running, output_text)
        """
        command = self.command_parser.parse(raw_text)

        if command == "empty":
            return True, None

        if command == "help":
            return True, HELP_TEXT

        if command == "reset":
            self.memory.reset()
            return True, "Conversation history cleared."

        if command == "exit":
            return False, "Goodbye."

        if command == "unknown":
            return True, "Unknown command. Type /help to see available commands."

        user_text = raw_text.strip()

        # Add the user message first so Ollama sees the current turn together with
        # the rest of the session history.
        self.memory.add_user_message(user_text)

        try:
            assistant_text = self.ollama_client.send_chat(
                model=self.config.model,
                messages=self.memory.get_messages(),
                temperature=self.config.temperature,
            )
        except OllamaServiceError as exc:
            # Remove the failed user turn so a broken request does not pollute the
            # in-memory conversation.
            self.memory.remove_last_message()
            return True, f"Error: {exc}"

        self.memory.add_assistant_message(assistant_text)
        return True, assistant_text

    def run_repl(self) -> int:
        """Run the interactive terminal loop until the user exits."""
        print("Type /help for commands.")

        while True:
            try:
                raw_text = input("\nYou: ")
            except KeyboardInterrupt:
                print("\n\nInterrupted. Exiting.")
                return 0
            except EOFError:
                print("\n\nInput closed. Exiting.")
                return 0

            keep_running, output = self.process_user_text(raw_text)

            if output:
                if output in {HELP_TEXT, "Conversation history cleared.", "Goodbye."}:
                    print(output)
                elif output.startswith("Unknown command") or output.startswith("Error:"):
                    print(output)
                else:
                    print(f"\nAssistant: {output}")

            if not keep_running:
                return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by both the terminal and the tests."""
    config = ConfigResolver.resolve(argv=argv)

    print(APP_NAME)
    print(f"Model: {config.model}")

    try:
        ollama_client = OllamaChatClient(host=config.ollama_host)
    except OllamaServiceError as exc:
        print("Ollama reachable: NO")
        print(f"Error: {exc}")
        return 1

    reachable, error_message = ollama_client.is_reachable()
    print(f"Ollama reachable: {'YES' if reachable else 'NO'}")

    if not reachable:
        print(f"Error: {error_message}")
        return 1

    app = Stage1CLIApp(config=config, ollama_client=ollama_client)
    return app.run_repl()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        # This is a second safety net in case Ctrl+C arrives outside input().
        print("\nInterrupted. Exiting.")
        raise SystemExit(0)