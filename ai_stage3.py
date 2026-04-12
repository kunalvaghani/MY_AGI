from __future__ import annotations

"""
Stage 3 local AI assistant.

This file preserves the Stage 1 chat loop and the Stage 2 transfer-learning
benchmark, then adds a bounded, inspectable Stage 3 rapid-adaptation mode.

Important honesty boundary:
- This is not model training.
- This is not persistent learning across runs.
- This is not Stage 4 few-shot / zero-shot competence.
- This is a small application-level adaptation loop that measures whether the
  model improves within the same run after a small correction or one/few
  corrective examples.
"""

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from ollama import Client, ResponseError
except ModuleNotFoundError as import_error:
    Client = None  # type: ignore[assignment]

    class ResponseError(Exception):
        """Fallback definition used only when the ollama package is missing."""

    OLLAMA_IMPORT_ERROR: ModuleNotFoundError | None = import_error
else:
    OLLAMA_IMPORT_ERROR = None


APP_NAME = "Local AI Assistant - Stage 3"
DEFAULT_MODEL = "gemma3:latest"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MODE = "chat"
DEFAULT_STAGE2_CASES_PATH = "stage2_transfer_cases.json"
DEFAULT_STAGE3_CASES_PATH = "stage3_adaptation_cases.json"
DEFAULT_STAGE2_SCORECARD_PATH = "stage2_scorecard.json"
DEFAULT_STAGE2_FAILURE_LOG_PATH = "failure_log_stage2.md"
DEFAULT_STAGE3_SCORECARD_PATH = "stage3_scorecard.json"
DEFAULT_STAGE3_FAILURE_LOG_PATH = "failure_log_stage3.md"

HELP_TEXT = """Available commands:
  /help   Show this help message
  /reset  Clear the current in-memory conversation
  /exit   Quit the application"""

SUPPORTED_MODES = [
    "chat",
    "transfer-demo",
    "transfer-eval",
    "adapt-demo",
    "adapt-eval",
]

ALLOWED_ADAPTATION_TYPES = {
    "explicit correction",
    "rule update",
    "one corrected example",
    "two corrected examples",
    "output format correction",
    "constraint correction",
}


@dataclass(frozen=True)
class AppConfig:
    """Resolved runtime configuration for the application."""

    model: str
    mode: str = DEFAULT_MODE
    cases_path: str = DEFAULT_STAGE2_CASES_PATH
    scorecard_out: str = DEFAULT_STAGE2_SCORECARD_PATH
    failure_log_out: str = DEFAULT_STAGE2_FAILURE_LOG_PATH
    limit: int | None = None
    ollama_host: str = DEFAULT_OLLAMA_HOST
    temperature: float = DEFAULT_TEMPERATURE


class ConfigResolver:
    """Resolve configuration from CLI arguments and environment variables."""

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=(
                "Run the Stage 3 local AI assistant against a local Ollama model. "
                "Modes: chat, transfer-demo, transfer-eval, adapt-demo, adapt-eval."
            )
        )
        parser.add_argument(
            "--mode",
            dest="mode",
            choices=SUPPORTED_MODES,
            default=DEFAULT_MODE,
            help="Run normal chat, a Stage 2 transfer mode, or a Stage 3 adaptation mode.",
        )
        parser.add_argument(
            "--model",
            dest="model",
            help="Override the Ollama model name for this run.",
        )
        parser.add_argument(
            "--cases",
            dest="cases_path",
            help="Path to the transfer or adaptation cases JSON file.",
        )
        parser.add_argument(
            "--limit",
            dest="limit",
            type=int,
            help="Limit the number of benchmark cases to run.",
        )
        parser.add_argument(
            "--scorecard-out",
            dest="scorecard_out",
            help="Path to write the scorecard JSON file.",
        )
        parser.add_argument(
            "--failure-log-out",
            dest="failure_log_out",
            help="Path to write the markdown failure log.",
        )
        return parser

    @classmethod
    def resolve(
        cls,
        argv: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> AppConfig:
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

        if args.limit is not None and args.limit <= 0:
            parser.error("--limit must be a positive integer when provided.")

        default_cases_path = cls._resolve_default_cases_path(args.mode)
        default_scorecard_out = cls._resolve_default_scorecard_out(args.mode)
        default_failure_log_out = cls._resolve_default_failure_log_out(args.mode)

        return AppConfig(
            model=selected_model,
            mode=args.mode,
            cases_path=(args.cases_path or default_cases_path),
            scorecard_out=(args.scorecard_out or default_scorecard_out),
            failure_log_out=(args.failure_log_out or default_failure_log_out),
            limit=args.limit,
        )

    @staticmethod
    def _resolve_default_cases_path(mode: str) -> str:
        if mode in {"adapt-demo", "adapt-eval"}:
            return DEFAULT_STAGE3_CASES_PATH
        return DEFAULT_STAGE2_CASES_PATH

    @staticmethod
    def _resolve_default_scorecard_out(mode: str) -> str:
        if mode in {"adapt-demo", "adapt-eval"}:
            return DEFAULT_STAGE3_SCORECARD_PATH
        return DEFAULT_STAGE2_SCORECARD_PATH

    @staticmethod
    def _resolve_default_failure_log_out(mode: str) -> str:
        if mode in {"adapt-demo", "adapt-eval"}:
            return DEFAULT_STAGE3_FAILURE_LOG_PATH
        return DEFAULT_STAGE2_FAILURE_LOG_PATH


class OllamaServiceError(Exception):
    """Raised when the app cannot talk to Ollama successfully."""


class OllamaChatClient:
    """Small wrapper around the official Ollama Python client."""

    def __init__(self, host: str) -> None:
        self.host = host

        if OLLAMA_IMPORT_ERROR is not None or Client is None:
            raise OllamaServiceError(
                "The 'ollama' Python package is not installed. "
                "Run: py -3.11 -m pip install -r requirements.txt"
            )

        self._client = Client(host=host)

    def is_reachable(self) -> tuple[bool, str | None]:
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
        message_obj = getattr(response, "message", None)
        if message_obj is not None:
            content = getattr(message_obj, "content", "")
            if isinstance(content, str):
                return content.strip()

        if isinstance(response, dict):
            message = response.get("message", {})
            if isinstance(message, dict):
                content = message.get("content", "")
                if isinstance(content, str):
                    return content.strip()

        raise OllamaServiceError("Could not read the text content from Ollama's response.")

    def _format_connection_error(self, exc: Exception) -> str:
        return (
            f"Could not connect to Ollama at {self.host}. "
            f"Make sure Ollama is installed and running. Details: {exc}"
        )


@dataclass
class SessionMemory:
    """In-memory conversation state for the current process only."""

    messages: list[dict[str, str]] = field(default_factory=list)

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})

    def reset(self) -> None:
        self.messages.clear()

    def remove_last_message(self) -> None:
        if self.messages:
            self.messages.pop()

    def get_messages(self) -> list[dict[str, str]]:
        return [message.copy() for message in self.messages]


class CommandParser:
    """Parse the small set of slash commands for chat mode."""

    def parse(self, raw_text: str) -> str:
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
    """Preserved Stage 1 chat runtime for normal chat mode."""

    def __init__(
        self,
        config: AppConfig,
        ollama_client: OllamaChatClient,
        memory: SessionMemory | None = None,
        command_parser: CommandParser | None = None,
    ) -> None:
        self.config = config
        self.ollama_client = ollama_client
        self.memory = memory if memory is not None else SessionMemory()
        self.command_parser = command_parser if command_parser is not None else CommandParser()

    def process_user_text(self, raw_text: str) -> tuple[bool, str | None]:
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
        self.memory.add_user_message(user_text)

        try:
            assistant_text = self.ollama_client.send_chat(
                model=self.config.model,
                messages=self.memory.get_messages(),
                temperature=self.config.temperature,
            )
        except OllamaServiceError as exc:
            self.memory.remove_last_message()
            return True, f"Error: {exc}"

        self.memory.add_assistant_message(assistant_text)
        return True, assistant_text

    def run_repl(self) -> int:
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


class Stage2CaseLoadError(Exception):
    """Raised when the transfer case file cannot be loaded safely."""


@dataclass(frozen=True)
class Stage2TransferCase:
    """One curated source->target transfer case."""

    case_id: str
    source_domain: str
    target_domain: str
    source_example_input: str
    source_example_output: str
    transfer_rule_summary: str
    target_input: str
    expected_answer: str
    scoring_type: str
    notes: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "Stage2TransferCase":
        required_fields = [
            "case_id",
            "source_domain",
            "target_domain",
            "source_example_input",
            "source_example_output",
            "transfer_rule_summary",
            "target_input",
            "expected_answer",
            "scoring_type",
            "notes",
        ]
        missing = [field_name for field_name in required_fields if field_name not in mapping]
        if missing:
            raise Stage2CaseLoadError(
                "Transfer case is missing required field(s): " + ", ".join(missing)
            )

        values: dict[str, str] = {}
        for field_name in required_fields:
            value = mapping[field_name]
            if not isinstance(value, str) or not value.strip():
                raise Stage2CaseLoadError(
                    f"Transfer case field '{field_name}' must be a non-empty string."
                )
            values[field_name] = value.strip()

        return cls(**values)


class TransferCaseLoader:
    """Load and validate the Stage 2 transfer benchmark cases."""

    def load(self, path: str, limit: int | None = None) -> list[Stage2TransferCase]:
        case_path = Path(path)
        if not case_path.exists():
            raise Stage2CaseLoadError(f"Transfer case file was not found: {case_path}")

        payload = self._read_json(case_path)

        if isinstance(payload, dict) and "cases" in payload:
            payload = payload["cases"]

        if not isinstance(payload, list):
            raise Stage2CaseLoadError(
                "Transfer case file must contain a JSON list of case objects "
                "or an object with a 'cases' list."
            )

        cases: list[Stage2TransferCase] = []
        seen_case_ids: set[str] = set()
        for item in payload:
            if not isinstance(item, dict):
                raise Stage2CaseLoadError(
                    "Each transfer case must be a JSON object with the required fields."
                )

            case = Stage2TransferCase.from_mapping(item)
            if case.case_id in seen_case_ids:
                raise Stage2CaseLoadError(f"Duplicate case_id found: {case.case_id}")
            seen_case_ids.add(case.case_id)
            cases.append(case)

        if limit is not None:
            return cases[:limit]
        return cases

    def _read_json(self, case_path: Path) -> Any:
        try:
            raw_text = case_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise Stage2CaseLoadError(f"Could not read transfer case file: {exc}") from exc

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise Stage2CaseLoadError(f"Transfer case file is not valid JSON: {exc}") from exc


class TransferPromptBuilder:
    """Build explicit, inspectable prompts for Stage 2."""

    def build_baseline_prompt(self, case: Stage2TransferCase) -> str:
        return (
            "You are being evaluated on a bounded Stage 2 transfer-learning benchmark.\n"
            "Solve the target task directly with no source example.\n"
            "Return only the final answer. Do not explain your reasoning.\n\n"
            f"Target domain: {case.target_domain}\n"
            "Target task:\n"
            f"{case.target_input}"
        )

    def build_transfer_prompt(self, case: Stage2TransferCase) -> str:
        return (
            "You are being evaluated on a bounded Stage 2 transfer-learning benchmark.\n"
            "Use the solved source example and the abstract rule below to solve the new target task.\n"
            "Return only the final answer. Do not explain your reasoning.\n\n"
            f"Source domain: {case.source_domain}\n"
            "Solved source example input:\n"
            f"{case.source_example_input}\n\n"
            "Solved source example output:\n"
            f"{case.source_example_output}\n\n"
            "Abstract transferable rule:\n"
            f"{case.transfer_rule_summary}\n\n"
            f"Target domain: {case.target_domain}\n"
            "Target task:\n"
            f"{case.target_input}\n\n"
            "Apply the same abstract pattern to the target task."
        )


@dataclass
class TransferCaseResult:
    """Structured result for one transfer benchmark case."""

    case_id: str
    source_domain: str
    target_domain: str
    scoring_type: str
    expected_answer: str
    baseline_answer: str
    transfer_assisted_answer: str
    baseline_pass: bool
    transfer_assisted_pass: bool
    improvement: bool
    regression: bool
    transfer_helped: bool
    baseline_scoring_error: str | None = None
    transfer_scoring_error: str | None = None
    probable_failure_reason: str | None = None


class ScoringEngine:
    """Deterministic scoring helpers shared across Stage 2 and Stage 3."""

    def score_answer(
        self,
        answer: str,
        expected_answer: str,
        scoring_type: str,
    ) -> tuple[bool, str | None]:
        scoring_key = self._normalize_scoring_type(scoring_type)
        try:
            if scoring_key == "exact":
                return answer.strip() == expected_answer.strip(), None

            if scoring_key == "case-insensitive exact":
                return answer.strip().casefold() == expected_answer.strip().casefold(), None

            if scoring_key == "contains":
                return expected_answer.strip().casefold() in answer.strip().casefold(), None

            if scoring_key == "regex":
                return re.search(expected_answer, answer, flags=re.DOTALL) is not None, None

            if scoring_key == "custom normalized exact":
                return self.normalize_whitespace(answer) == self.normalize_whitespace(expected_answer), None

            return False, f"Unknown scoring_type '{scoring_type}'."
        except re.error as exc:
            return False, f"Invalid regex scoring pattern: {exc}"
        except Exception as exc:  # pragma: no cover - defensive guard
            return False, f"Scoring failed unexpectedly: {exc}"

    def normalize_whitespace(self, value: str) -> str:
        return " ".join(value.split())

    def _normalize_scoring_type(self, scoring_type: str) -> str:
        key = " ".join(scoring_type.strip().lower().replace("_", " ").split())
        if key == "normalized exact":
            return "custom normalized exact"
        return key


class TransferEvaluator:
    """Run baseline vs transfer-assisted evaluation for Stage 2."""

    def __init__(
        self,
        ollama_client: OllamaChatClient,
        model: str,
        temperature: float,
        prompt_builder: TransferPromptBuilder | None = None,
        scoring_engine: ScoringEngine | None = None,
    ) -> None:
        self.ollama_client = ollama_client
        self.model = model
        self.temperature = temperature
        self.prompt_builder = prompt_builder if prompt_builder is not None else TransferPromptBuilder()
        self.scoring_engine = scoring_engine if scoring_engine is not None else ScoringEngine()

    def evaluate_case(self, case: Stage2TransferCase) -> TransferCaseResult:
        baseline_answer = self._run_single_prompt(self.prompt_builder.build_baseline_prompt(case))
        transfer_answer = self._run_single_prompt(self.prompt_builder.build_transfer_prompt(case))

        baseline_pass, baseline_scoring_error = self.scoring_engine.score_answer(
            answer=baseline_answer,
            expected_answer=case.expected_answer,
            scoring_type=case.scoring_type,
        )
        transfer_pass, transfer_scoring_error = self.scoring_engine.score_answer(
            answer=transfer_answer,
            expected_answer=case.expected_answer,
            scoring_type=case.scoring_type,
        )

        improvement = (not baseline_pass) and transfer_pass
        regression = baseline_pass and (not transfer_pass)
        probable_failure_reason = self._infer_failure_reason(
            baseline_answer=baseline_answer,
            transfer_answer=transfer_answer,
            baseline_pass=baseline_pass,
            transfer_pass=transfer_pass,
            baseline_scoring_error=baseline_scoring_error,
            transfer_scoring_error=transfer_scoring_error,
        )

        return TransferCaseResult(
            case_id=case.case_id,
            source_domain=case.source_domain,
            target_domain=case.target_domain,
            scoring_type=case.scoring_type,
            expected_answer=case.expected_answer,
            baseline_answer=baseline_answer,
            transfer_assisted_answer=transfer_answer,
            baseline_pass=baseline_pass,
            transfer_assisted_pass=transfer_pass,
            improvement=improvement,
            regression=regression,
            transfer_helped=improvement,
            baseline_scoring_error=baseline_scoring_error,
            transfer_scoring_error=transfer_scoring_error,
            probable_failure_reason=probable_failure_reason,
        )

    def evaluate_cases(self, cases: Sequence[Stage2TransferCase]) -> list[TransferCaseResult]:
        return [self.evaluate_case(case) for case in cases]

    def build_scorecard(self, results: Sequence[TransferCaseResult]) -> dict[str, Any]:
        total_cases = len(results)
        baseline_pass_count = sum(result.baseline_pass for result in results)
        transfer_pass_count = sum(result.transfer_assisted_pass for result in results)
        improvement_count = sum(result.improvement for result in results)
        regression_count = sum(result.regression for result in results)

        baseline_pass_rate = (baseline_pass_count / total_cases) if total_cases else 0.0
        transfer_pass_rate = (transfer_pass_count / total_cases) if total_cases else 0.0

        return {
            "total_cases": total_cases,
            "baseline_pass_count": baseline_pass_count,
            "transfer_assisted_pass_count": transfer_pass_count,
            "improvement_count": improvement_count,
            "regression_count": regression_count,
            "baseline_pass_rate": round(baseline_pass_rate, 4),
            "pass_rate": round(transfer_pass_rate, 4),
            "transfer_assisted_pass_rate": round(transfer_pass_rate, 4),
            "per_case_details": [asdict(result) for result in results],
        }

    def _run_single_prompt(self, prompt: str) -> str:
        try:
            return self.ollama_client.send_chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
        except OllamaServiceError as exc:
            return f"ERROR: {exc}"

    def _infer_failure_reason(
        self,
        baseline_answer: str,
        transfer_answer: str,
        baseline_pass: bool,
        transfer_pass: bool,
        baseline_scoring_error: str | None,
        transfer_scoring_error: str | None,
    ) -> str | None:
        if baseline_pass and transfer_pass:
            return None

        if baseline_scoring_error or transfer_scoring_error:
            problems = [item for item in [baseline_scoring_error, transfer_scoring_error] if item]
            return " ; ".join(problems)

        if baseline_answer.startswith("ERROR:") or transfer_answer.startswith("ERROR:"):
            return "A model call failed during evaluation."

        if baseline_pass and not transfer_pass:
            return "Transfer prompt likely distracted the model or overfit the source example."

        if (not baseline_pass) and (not transfer_pass):
            baseline_norm = self.scoring_engine.normalize_whitespace(baseline_answer).casefold()
            transfer_norm = self.scoring_engine.normalize_whitespace(transfer_answer).casefold()
            if baseline_norm == transfer_norm:
                return "Transfer prompt did not materially change the answer."
            return "Abstract rule was not applied correctly to the target task."

        return None


class Stage3CaseLoadError(Exception):
    """Raised when the adaptation case file cannot be loaded safely."""


@dataclass(frozen=True)
class Stage3AdaptationCase:
    """One curated Stage 3 rapid-adaptation case."""

    case_id: str
    domain: str
    adaptation_type: str
    initial_task: str
    expected_initial_answer: str
    feedback_type: str
    feedback_payload: str
    followup_task: str
    expected_followup_answer: str
    scoring_type: str
    notes: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "Stage3AdaptationCase":
        required_fields = [
            "case_id",
            "domain",
            "adaptation_type",
            "initial_task",
            "expected_initial_answer",
            "feedback_type",
            "feedback_payload",
            "followup_task",
            "expected_followup_answer",
            "scoring_type",
            "notes",
        ]
        missing = [field_name for field_name in required_fields if field_name not in mapping]
        if missing:
            raise Stage3CaseLoadError(
                "Adaptation case is missing required field(s): " + ", ".join(missing)
            )

        values: dict[str, str] = {}
        for field_name in required_fields:
            value = mapping[field_name]
            if not isinstance(value, str) or not value.strip():
                raise Stage3CaseLoadError(
                    f"Adaptation case field '{field_name}' must be a non-empty string."
                )
            values[field_name] = value.strip()

        adaptation_type = values["adaptation_type"].lower()
        if adaptation_type not in ALLOWED_ADAPTATION_TYPES:
            allowed = ", ".join(sorted(ALLOWED_ADAPTATION_TYPES))
            raise Stage3CaseLoadError(
                f"Unsupported adaptation_type '{values['adaptation_type']}'. Allowed values: {allowed}."
            )

        return cls(**values)


class AdaptationCaseLoader:
    """Load and validate the Stage 3 adaptation benchmark cases."""

    def load(self, path: str, limit: int | None = None) -> list[Stage3AdaptationCase]:
        case_path = Path(path)
        if not case_path.exists():
            raise Stage3CaseLoadError(f"Adaptation case file was not found: {case_path}")

        payload = self._read_json(case_path)

        if isinstance(payload, dict) and "cases" in payload:
            payload = payload["cases"]

        if not isinstance(payload, list):
            raise Stage3CaseLoadError(
                "Adaptation case file must contain a JSON list of case objects "
                "or an object with a 'cases' list."
            )

        cases: list[Stage3AdaptationCase] = []
        seen_case_ids: set[str] = set()
        for item in payload:
            if not isinstance(item, dict):
                raise Stage3CaseLoadError(
                    "Each adaptation case must be a JSON object with the required fields."
                )

            case = Stage3AdaptationCase.from_mapping(item)
            if case.case_id in seen_case_ids:
                raise Stage3CaseLoadError(f"Duplicate case_id found: {case.case_id}")
            seen_case_ids.add(case.case_id)
            cases.append(case)

        if limit is not None:
            return cases[:limit]
        return cases

    def _read_json(self, case_path: Path) -> Any:
        try:
            raw_text = case_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise Stage3CaseLoadError(f"Could not read adaptation case file: {exc}") from exc

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise Stage3CaseLoadError(f"Adaptation case file is not valid JSON: {exc}") from exc


class AdaptationPromptBuilder:
    """Build explicit, inspectable prompts for Stage 3 rapid adaptation."""

    def build_initial_prompt(self, case: Stage3AdaptationCase) -> str:
        return (
            "You are being evaluated on a bounded Stage 3 rapid-adaptation benchmark.\n"
            "This is the initial attempt, before any correction is provided.\n"
            "Solve the task as best you can.\n"
            "Return only the final answer. Do not explain your reasoning.\n\n"
            f"Domain: {case.domain}\n"
            f"Adaptation type: {case.adaptation_type}\n"
            "Initial task:\n"
            f"{case.initial_task}"
        )

    def build_adapted_prompt(self, case: Stage3AdaptationCase, initial_answer: str) -> str:
        return (
            "You are being evaluated on a bounded Stage 3 rapid-adaptation benchmark.\n"
            "This is the adapted attempt after explicit feedback or corrected examples.\n"
            "Apply the correction or updated rule to the follow-up task.\n"
            "Return only the final answer. Do not explain your reasoning.\n\n"
            f"Domain: {case.domain}\n"
            f"Adaptation type: {case.adaptation_type}\n"
            f"Feedback type: {case.feedback_type}\n\n"
            "Initial task:\n"
            f"{case.initial_task}\n\n"
            "Model's first answer:\n"
            f"{initial_answer}\n\n"
            "Feedback received:\n"
            f"{case.feedback_payload}\n\n"
            "Follow-up task:\n"
            f"{case.followup_task}\n\n"
            "Apply the feedback exactly when answering the follow-up task."
        )


@dataclass
class AdaptationCaseResult:
    """Structured result for one Stage 3 adaptation case."""

    case_id: str
    domain: str
    adaptation_type: str
    feedback_type: str
    feedback_payload: str
    scoring_type: str
    initial_answer: str
    adapted_answer: str
    expected_initial_answer: str
    expected_followup_answer: str
    initial_pass: bool
    adapted_pass: bool
    adaptation_helped: bool
    regression: bool
    scoring_error_initial: str | None = None
    scoring_error_adapted: str | None = None
    probable_failure_reason: str | None = None


class AdaptationEvaluator:
    """Run inspectable before/after adaptation evaluation for Stage 3."""

    def __init__(
        self,
        ollama_client: OllamaChatClient,
        model: str,
        temperature: float,
        prompt_builder: AdaptationPromptBuilder | None = None,
        scoring_engine: ScoringEngine | None = None,
    ) -> None:
        self.ollama_client = ollama_client
        self.model = model
        self.temperature = temperature
        self.prompt_builder = prompt_builder if prompt_builder is not None else AdaptationPromptBuilder()
        self.scoring_engine = scoring_engine if scoring_engine is not None else ScoringEngine()

    def evaluate_case(self, case: Stage3AdaptationCase) -> AdaptationCaseResult:
        initial_answer = self._run_single_prompt(self.prompt_builder.build_initial_prompt(case))
        adapted_answer = self._run_single_prompt(
            self.prompt_builder.build_adapted_prompt(case=case, initial_answer=initial_answer)
        )

        initial_pass, initial_scoring_error = self.scoring_engine.score_answer(
            answer=initial_answer,
            expected_answer=case.expected_initial_answer,
            scoring_type=case.scoring_type,
        )
        adapted_pass, adapted_scoring_error = self.scoring_engine.score_answer(
            answer=adapted_answer,
            expected_answer=case.expected_followup_answer,
            scoring_type=case.scoring_type,
        )

        adaptation_helped = (not initial_pass) and adapted_pass
        regression = initial_pass and (not adapted_pass)
        probable_failure_reason = self._infer_failure_reason(
            initial_answer=initial_answer,
            adapted_answer=adapted_answer,
            initial_pass=initial_pass,
            adapted_pass=adapted_pass,
            initial_scoring_error=initial_scoring_error,
            adapted_scoring_error=adapted_scoring_error,
        )

        return AdaptationCaseResult(
            case_id=case.case_id,
            domain=case.domain,
            adaptation_type=case.adaptation_type,
            feedback_type=case.feedback_type,
            feedback_payload=case.feedback_payload,
            scoring_type=case.scoring_type,
            initial_answer=initial_answer,
            adapted_answer=adapted_answer,
            expected_initial_answer=case.expected_initial_answer,
            expected_followup_answer=case.expected_followup_answer,
            initial_pass=initial_pass,
            adapted_pass=adapted_pass,
            adaptation_helped=adaptation_helped,
            regression=regression,
            scoring_error_initial=initial_scoring_error,
            scoring_error_adapted=adapted_scoring_error,
            probable_failure_reason=probable_failure_reason,
        )

    def evaluate_cases(self, cases: Sequence[Stage3AdaptationCase]) -> list[AdaptationCaseResult]:
        return [self.evaluate_case(case) for case in cases]

    def build_scorecard(self, results: Sequence[AdaptationCaseResult]) -> dict[str, Any]:
        total_cases = len(results)
        initial_pass_count = sum(result.initial_pass for result in results)
        adapted_pass_count = sum(result.adapted_pass for result in results)
        adaptation_success_count = sum(result.adaptation_helped for result in results)
        regression_count = sum(result.regression for result in results)

        initial_pass_rate = (initial_pass_count / total_cases) if total_cases else 0.0
        adapted_pass_rate = (adapted_pass_count / total_cases) if total_cases else 0.0

        return {
            "total_cases": total_cases,
            "initial_pass_count": initial_pass_count,
            "adapted_pass_count": adapted_pass_count,
            "adaptation_success_count": adaptation_success_count,
            "regression_count": regression_count,
            "initial_pass_rate": round(initial_pass_rate, 4),
            "pass_rate": round(adapted_pass_rate, 4),
            "adapted_pass_rate": round(adapted_pass_rate, 4),
            "per_case_details": [asdict(result) for result in results],
        }

    def _run_single_prompt(self, prompt: str) -> str:
        try:
            return self.ollama_client.send_chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
        except OllamaServiceError as exc:
            return f"ERROR: {exc}"

    def _infer_failure_reason(
        self,
        initial_answer: str,
        adapted_answer: str,
        initial_pass: bool,
        adapted_pass: bool,
        initial_scoring_error: str | None,
        adapted_scoring_error: str | None,
    ) -> str | None:
        if initial_pass and adapted_pass:
            return None

        if initial_scoring_error or adapted_scoring_error:
            problems = [item for item in [initial_scoring_error, adapted_scoring_error] if item]
            return " ; ".join(problems)

        if initial_answer.startswith("ERROR:") or adapted_answer.startswith("ERROR:"):
            return "A model call failed during evaluation."

        if initial_pass and not adapted_pass:
            return "The correction step appears to have destabilized a previously correct behavior."

        if (not initial_pass) and (not adapted_pass):
            initial_norm = self.scoring_engine.normalize_whitespace(initial_answer).casefold()
            adapted_norm = self.scoring_engine.normalize_whitespace(adapted_answer).casefold()
            if initial_norm == adapted_norm:
                return "Feedback did not materially change the answer."
            return "The follow-up answer changed, but the correction was still not applied correctly."

        return None


class ScorecardWriter:
    """Write a structured JSON scorecard to disk."""

    def write(self, path: str | Path, scorecard: Mapping[str, Any]) -> Path:
        output_path = Path(path)
        output_path.write_text(json.dumps(scorecard, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path


class FailureLogWriter:
    """Write human-readable markdown failure logs."""

    def write_transfer_log(self, path: str | Path, results: Sequence[TransferCaseResult]) -> Path:
        output_path = Path(path)
        failed_results = [result for result in results if not result.transfer_assisted_pass]

        lines: list[str] = [
            "# Stage 2 Failure Log",
            "",
            f"Total failed transfer-assisted cases: {len(failed_results)}",
            "",
        ]

        if not failed_results:
            lines.extend(
                [
                    "All transfer-assisted cases passed in this run.",
                    "",
                    "No failure entries were generated.",
                ]
            )
        else:
            for result in failed_results:
                lines.extend(
                    [
                        f"## {result.case_id}",
                        "",
                        f"- Expected answer: `{result.expected_answer}`",
                        f"- Baseline answer: `{result.baseline_answer}`",
                        f"- Transfer-assisted answer: `{result.transfer_assisted_answer}`",
                        f"- Probable failure reason: {result.probable_failure_reason or 'Unknown'}",
                        "",
                    ]
                )

        output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return output_path

    def write_adaptation_log(self, path: str | Path, results: Sequence[AdaptationCaseResult]) -> Path:
        output_path = Path(path)
        failed_results = [result for result in results if not result.adapted_pass]

        lines: list[str] = [
            "# Stage 3 Failure Log",
            "",
            f"Total failed adapted cases: {len(failed_results)}",
            "",
        ]

        if not failed_results:
            lines.extend(
                [
                    "All adapted follow-up cases passed in this run.",
                    "",
                    "No failure entries were generated.",
                ]
            )
        else:
            for result in failed_results:
                lines.extend(
                    [
                        f"## {result.case_id}",
                        "",
                        f"- Expected initial answer: `{result.expected_initial_answer}`",
                        f"- Initial answer: `{result.initial_answer}`",
                        f"- Feedback received: `{result.feedback_payload}`",
                        f"- Expected follow-up answer: `{result.expected_followup_answer}`",
                        f"- Adapted answer: `{result.adapted_answer}`",
                        f"- Probable failure reason: {result.probable_failure_reason or 'Unknown'}",
                        "",
                    ]
                )

        output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return output_path


class Stage3CLIApp:
    """Top-level controller preserving Stage 1/2 and adding Stage 3 modes."""

    def __init__(
        self,
        config: AppConfig,
        ollama_client: OllamaChatClient,
        transfer_case_loader: TransferCaseLoader | None = None,
        adaptation_case_loader: AdaptationCaseLoader | None = None,
        scorecard_writer: ScorecardWriter | None = None,
        failure_log_writer: FailureLogWriter | None = None,
    ) -> None:
        self.config = config
        self.ollama_client = ollama_client
        self.transfer_case_loader = (
            transfer_case_loader if transfer_case_loader is not None else TransferCaseLoader()
        )
        self.adaptation_case_loader = (
            adaptation_case_loader if adaptation_case_loader is not None else AdaptationCaseLoader()
        )
        self.scorecard_writer = scorecard_writer if scorecard_writer is not None else ScorecardWriter()
        self.failure_log_writer = (
            failure_log_writer if failure_log_writer is not None else FailureLogWriter()
        )

    def run(self) -> int:
        if self.config.mode == "chat":
            chat_app = Stage1CLIApp(config=self.config, ollama_client=self.ollama_client)
            return chat_app.run_repl()

        if self.config.mode in {"transfer-demo", "transfer-eval"}:
            return self._run_transfer_modes()

        if self.config.mode in {"adapt-demo", "adapt-eval"}:
            return self._run_adaptation_modes()

        print(f"Error: Unsupported mode '{self.config.mode}'.")
        return 1

    def _run_transfer_modes(self) -> int:
        try:
            cases = self.transfer_case_loader.load(self.config.cases_path, limit=self.config.limit)
        except Stage2CaseLoadError as exc:
            print(f"Error: {exc}")
            return 1

        if not cases:
            print("Error: The transfer case file loaded successfully but contained no cases.")
            return 1

        evaluator = TransferEvaluator(
            ollama_client=self.ollama_client,
            model=self.config.model,
            temperature=self.config.temperature,
        )

        if self.config.mode == "transfer-demo":
            return self._run_transfer_demo(evaluator=evaluator, cases=cases)

        return self._run_transfer_eval(evaluator=evaluator, cases=cases)

    def _run_adaptation_modes(self) -> int:
        try:
            cases = self.adaptation_case_loader.load(self.config.cases_path, limit=self.config.limit)
        except Stage3CaseLoadError as exc:
            print(f"Error: {exc}")
            return 1

        if not cases:
            print("Error: The adaptation case file loaded successfully but contained no cases.")
            return 1

        evaluator = AdaptationEvaluator(
            ollama_client=self.ollama_client,
            model=self.config.model,
            temperature=self.config.temperature,
        )

        if self.config.mode == "adapt-demo":
            return self._run_adapt_demo(evaluator=evaluator, cases=cases)

        return self._run_adapt_eval(evaluator=evaluator, cases=cases)

    def _run_transfer_demo(
        self,
        evaluator: TransferEvaluator,
        cases: Sequence[Stage2TransferCase],
    ) -> int:
        print(f"Loaded transfer cases: {len(cases)}")

        for index, case in enumerate(cases, start=1):
            result = evaluator.evaluate_case(case)
            print("-" * 72)
            print(f"Demo case {index}: {case.case_id}")
            print(f"Source domain: {case.source_domain}")
            print(f"Target domain: {case.target_domain}")
            print(f"Baseline answer: {result.baseline_answer}")
            print(f"Transfer-assisted answer: {result.transfer_assisted_answer}")
            print(f"Expected answer: {case.expected_answer}")
            print(f"Transfer helped: {'YES' if result.transfer_helped else 'NO'}")

        return 0

    def _run_transfer_eval(
        self,
        evaluator: TransferEvaluator,
        cases: Sequence[Stage2TransferCase],
    ) -> int:
        results = evaluator.evaluate_cases(cases)
        scorecard = evaluator.build_scorecard(results)

        scorecard_path = self.scorecard_writer.write(self.config.scorecard_out, scorecard)
        failure_log_path = self.failure_log_writer.write_transfer_log(self.config.failure_log_out, results)

        print("Transfer evaluation complete.")
        print(f"Total cases: {scorecard['total_cases']}")
        print(f"Baseline passes: {scorecard['baseline_pass_count']}")
        print(f"Transfer-assisted passes: {scorecard['transfer_assisted_pass_count']}")
        print(f"Improvements: {scorecard['improvement_count']}")
        print(f"Regressions: {scorecard['regression_count']}")
        print(f"Baseline pass rate: {scorecard['baseline_pass_rate']:.2%}")
        print(f"Transfer-assisted pass rate: {scorecard['transfer_assisted_pass_rate']:.2%}")
        print(f"Scorecard written to: {scorecard_path}")
        print(f"Failure log written to: {failure_log_path}")

        return 0

    def _run_adapt_demo(
        self,
        evaluator: AdaptationEvaluator,
        cases: Sequence[Stage3AdaptationCase],
    ) -> int:
        print(f"Loaded adaptation cases: {len(cases)}")

        for index, case in enumerate(cases, start=1):
            result = evaluator.evaluate_case(case)
            print("-" * 72)
            print(f"Demo case {index}: {case.case_id}")
            print(f"Domain: {case.domain}")
            print(f"Adaptation type: {case.adaptation_type}")
            print("Initial task / rule set:")
            print(case.initial_task)
            print(f"Initial answer: {result.initial_answer}")
            print("Feedback or correction received:")
            print(case.feedback_payload)
            print("Follow-up task:")
            print(case.followup_task)
            print(f"Adapted answer: {result.adapted_answer}")
            print(f"Expected answer: {case.expected_followup_answer}")
            print(f"Adaptation helped: {'YES' if result.adaptation_helped else 'NO'}")

        return 0

    def _run_adapt_eval(
        self,
        evaluator: AdaptationEvaluator,
        cases: Sequence[Stage3AdaptationCase],
    ) -> int:
        results = evaluator.evaluate_cases(cases)
        scorecard = evaluator.build_scorecard(results)

        scorecard_path = self.scorecard_writer.write(self.config.scorecard_out, scorecard)
        failure_log_path = self.failure_log_writer.write_adaptation_log(
            self.config.failure_log_out,
            results,
        )

        print("Adaptation evaluation complete.")
        print(f"Total cases: {scorecard['total_cases']}")
        print(f"Initial passes: {scorecard['initial_pass_count']}")
        print(f"Adapted passes: {scorecard['adapted_pass_count']}")
        print(f"Adaptation successes: {scorecard['adaptation_success_count']}")
        print(f"Regressions: {scorecard['regression_count']}")
        print(f"Initial pass rate: {scorecard['initial_pass_rate']:.2%}")
        print(f"Adapted pass rate: {scorecard['adapted_pass_rate']:.2%}")
        print(f"Scorecard written to: {scorecard_path}")
        print(f"Failure log written to: {failure_log_path}")

        return 0


def main(argv: Sequence[str] | None = None) -> int:
    config = ConfigResolver.resolve(argv=argv)

    print(APP_NAME)
    print(f"Model: {config.model}")
    print(f"Mode: {config.mode}")

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

    app = Stage3CLIApp(config=config, ollama_client=ollama_client)
    return app.run()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
        raise SystemExit(0)
