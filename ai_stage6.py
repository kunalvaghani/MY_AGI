
from __future__ import annotations

"""
Stage 6 local AI assistant.

This file preserves the Stage 1 chat loop, the Stage 2 transfer-learning
benchmark, the Stage 3 rapid-adaptation benchmark, and the Stage 4 few-shot /
zero-shot benchmark, then adds a bounded, inspectable Stage 5 robust
reasoning mode and a separate Stage 6 common-sense understanding mode.

Important honesty boundary:
- This is not model training.
- This is not persistent learning across runs.
- This is not hidden long-term learning.
- This is not Stage 4 few-shot / zero-shot competence.
- This is not Stage 9 long-horizon planning.
- This is not Stage 11 subgoal decomposition.
- This is not model training or persistent learning.
- Stage 6 is a bounded common-sense benchmark about everyday implied facts.
- It does not claim world modeling, social intelligence, or hidden long-term learning.
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


APP_NAME = "Local AI Assistant - Stage 6"
DEFAULT_MODEL = "gemma3:latest"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MODE = "chat"
DEFAULT_STAGE2_CASES_PATH = "stage2_transfer_cases.json"
DEFAULT_STAGE3_CASES_PATH = "stage3_adaptation_cases.json"
DEFAULT_STAGE4_CASES_PATH = "stage4_fewshot_cases.json"
DEFAULT_STAGE5_CASES_PATH = "stage5_reasoning_cases.json"
DEFAULT_STAGE6_CASES_PATH = "stage6_commonsense_cases.json"
DEFAULT_STAGE2_SCORECARD_PATH = "stage2_scorecard.json"
DEFAULT_STAGE3_SCORECARD_PATH = "stage3_scorecard.json"
DEFAULT_STAGE4_SCORECARD_PATH = "stage4_scorecard.json"
DEFAULT_STAGE5_SCORECARD_PATH = "stage5_scorecard.json"
DEFAULT_STAGE6_SCORECARD_PATH = "stage6_scorecard.json"
DEFAULT_STAGE2_FAILURE_LOG_PATH = "failure_log_stage2.md"
DEFAULT_STAGE3_FAILURE_LOG_PATH = "failure_log_stage3.md"
DEFAULT_STAGE4_FAILURE_LOG_PATH = "failure_log_stage4.md"
DEFAULT_STAGE5_FAILURE_LOG_PATH = "failure_log_stage5.md"
DEFAULT_STAGE6_FAILURE_LOG_PATH = "failure_log_stage6.md"

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
    "fewshot-demo",
    "fewshot-eval",
    "reason-demo",
    "reason-eval",
    "commonsense-demo",
    "commonsense-eval",
]

ALLOWED_ADAPTATION_TYPES = {
    "explicit correction",
    "rule update",
    "one corrected example",
    "two corrected examples",
    "output format correction",
    "constraint correction",
}

ALLOWED_REASONING_SCORING_TYPES = {
    "exact",
    "case-insensitive exact",
    "contains",
    "regex",
    "custom normalized exact",
    "normalized exact",
}

ALLOWED_COMMONSENSE_SCORING_TYPES = ALLOWED_REASONING_SCORING_TYPES


def read_json_payload(path: str | Path, label: str) -> Any:
    """Read and parse a JSON file with a consistent error surface."""

    file_path = Path(path)
    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read {label} file: {exc}") from exc

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label.capitalize()} file is not valid JSON: {exc}") from exc


def require_non_empty_string(
    mapping: Mapping[str, Any],
    field_name: str,
    error_type: type[Exception],
    object_label: str,
) -> str:
    """Validate a required non-empty string field."""

    value = mapping.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"{object_label} field '{field_name}' must be a non-empty string.")
    return value.strip()


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
                "Run the Stage 6 local AI assistant against a local Ollama model. "
                "Modes: chat, transfer-demo, transfer-eval, adapt-demo, adapt-eval, "
                "fewshot-demo, fewshot-eval, reason-demo, reason-eval, commonsense-demo, commonsense-eval."
            )
        )
        parser.add_argument(
            "--mode",
            dest="mode",
            choices=SUPPORTED_MODES,
            default=DEFAULT_MODE,
            help="Run normal chat, Stage 2 transfer modes, Stage 3 adaptation modes, Stage 4 few-shot modes, Stage 5 reasoning modes, or Stage 6 commonsense modes.",
        )
        parser.add_argument(
            "--model",
            dest="model",
            help="Override the Ollama model name for this run.",
        )
        parser.add_argument(
            "--cases",
            dest="cases_path",
            help="Path to the active benchmark case file.",
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

        return AppConfig(
            model=selected_model,
            mode=args.mode,
            cases_path=(args.cases_path or cls._resolve_default_cases_path(args.mode)),
            scorecard_out=(args.scorecard_out or cls._resolve_default_scorecard_out(args.mode)),
            failure_log_out=(args.failure_log_out or cls._resolve_default_failure_log_out(args.mode)),
            limit=args.limit,
        )

    @staticmethod
    def _resolve_default_cases_path(mode: str) -> str:
        if mode in {"adapt-demo", "adapt-eval"}:
            return DEFAULT_STAGE3_CASES_PATH
        if mode in {"fewshot-demo", "fewshot-eval"}:
            return DEFAULT_STAGE4_CASES_PATH
        if mode in {"reason-demo", "reason-eval"}:
            return DEFAULT_STAGE5_CASES_PATH
        if mode in {"commonsense-demo", "commonsense-eval"}:
            return DEFAULT_STAGE6_CASES_PATH
        return DEFAULT_STAGE2_CASES_PATH

    @staticmethod
    def _resolve_default_scorecard_out(mode: str) -> str:
        if mode in {"adapt-demo", "adapt-eval"}:
            return DEFAULT_STAGE3_SCORECARD_PATH
        if mode in {"fewshot-demo", "fewshot-eval"}:
            return DEFAULT_STAGE4_SCORECARD_PATH
        if mode in {"reason-demo", "reason-eval"}:
            return DEFAULT_STAGE5_SCORECARD_PATH
        if mode in {"commonsense-demo", "commonsense-eval"}:
            return DEFAULT_STAGE6_SCORECARD_PATH
        return DEFAULT_STAGE2_SCORECARD_PATH

    @staticmethod
    def _resolve_default_failure_log_out(mode: str) -> str:
        if mode in {"adapt-demo", "adapt-eval"}:
            return DEFAULT_STAGE3_FAILURE_LOG_PATH
        if mode in {"fewshot-demo", "fewshot-eval"}:
            return DEFAULT_STAGE4_FAILURE_LOG_PATH
        if mode in {"reason-demo", "reason-eval"}:
            return DEFAULT_STAGE5_FAILURE_LOG_PATH
        if mode in {"commonsense-demo", "commonsense-eval"}:
            return DEFAULT_STAGE6_FAILURE_LOG_PATH
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
                    f"Model '{model}' was not found in Ollama. Pull it first with: ollama pull {model}"
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
    """One curated source-to-target transfer case."""

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
            values[field_name] = require_non_empty_string(
                mapping=mapping,
                field_name=field_name,
                error_type=Stage2CaseLoadError,
                object_label="Transfer case",
            )

        return cls(**values)


class TransferCaseLoader:
    """Load and validate the Stage 2 transfer benchmark cases."""

    def load(self, path: str, limit: int | None = None) -> list[Stage2TransferCase]:
        case_path = Path(path)
        if not case_path.exists():
            raise Stage2CaseLoadError(f"Transfer case file was not found: {case_path}")

        try:
            payload = read_json_payload(case_path, "transfer case")
        except ValueError as exc:
            raise Stage2CaseLoadError(str(exc)) from exc

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

        return cases[:limit] if limit is not None else cases


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
    """Deterministic scoring helpers shared across all benchmark stages."""

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
    """Run baseline versus transfer-assisted evaluation for Stage 2."""

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
            values[field_name] = require_non_empty_string(
                mapping=mapping,
                field_name=field_name,
                error_type=Stage3CaseLoadError,
                object_label="Adaptation case",
            )

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

        try:
            payload = read_json_payload(case_path, "adaptation case")
        except ValueError as exc:
            raise Stage3CaseLoadError(str(exc)) from exc

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

        return cases[:limit] if limit is not None else cases


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


class Stage4CaseLoadError(Exception):
    """Raised when the few-shot case file cannot be loaded safely."""


@dataclass(frozen=True)
class FewShotExample:
    """One tiny demonstration used for a Stage 4 few-shot prompt."""

    example_input: str
    example_output: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "FewShotExample":
        if not isinstance(mapping, Mapping):
            raise Stage4CaseLoadError(
                "Each few-shot example must be a JSON object with 'example_input' and 'example_output'."
            )

        example_input = require_non_empty_string(
            mapping=mapping,
            field_name="example_input",
            error_type=Stage4CaseLoadError,
            object_label="Few-shot example",
        )
        example_output = require_non_empty_string(
            mapping=mapping,
            field_name="example_output",
            error_type=Stage4CaseLoadError,
            object_label="Few-shot example",
        )
        return cls(example_input=example_input, example_output=example_output)


@dataclass(frozen=True)
class Stage4FewShotCase:
    """One curated Stage 4 few-shot / zero-shot competence case."""

    case_id: str
    task_family: str
    task_description: str
    zero_shot_instruction: str
    few_shot_examples: list[FewShotExample]
    target_input: str
    expected_answer: str
    scoring_type: str
    novelty_notes: str
    notes: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "Stage4FewShotCase":
        required_string_fields = [
            "case_id",
            "task_family",
            "task_description",
            "zero_shot_instruction",
            "target_input",
            "expected_answer",
            "scoring_type",
            "novelty_notes",
            "notes",
        ]
        missing = [field_name for field_name in required_string_fields + ["few_shot_examples"] if field_name not in mapping]
        if missing:
            raise Stage4CaseLoadError(
                "Few-shot case is missing required field(s): " + ", ".join(missing)
            )

        values: dict[str, str] = {}
        for field_name in required_string_fields:
            values[field_name] = require_non_empty_string(
                mapping=mapping,
                field_name=field_name,
                error_type=Stage4CaseLoadError,
                object_label="Few-shot case",
            )

        raw_examples = mapping["few_shot_examples"]
        if not isinstance(raw_examples, list):
            raise Stage4CaseLoadError("Few-shot case field 'few_shot_examples' must be a JSON list.")
        if not 1 <= len(raw_examples) <= 3:
            raise Stage4CaseLoadError(
                "Few-shot case field 'few_shot_examples' must contain between 1 and 3 demonstrations."
            )

        examples = [FewShotExample.from_mapping(item) for item in raw_examples]
        return cls(few_shot_examples=examples, **values)


class FewShotCaseLoader:
    """Load and validate the Stage 4 few-shot / zero-shot benchmark cases."""

    def load(self, path: str, limit: int | None = None) -> list[Stage4FewShotCase]:
        case_path = Path(path)
        if not case_path.exists():
            raise Stage4CaseLoadError(f"Few-shot case file was not found: {case_path}")

        try:
            payload = read_json_payload(case_path, "few-shot case")
        except ValueError as exc:
            raise Stage4CaseLoadError(str(exc)) from exc

        if isinstance(payload, dict) and "cases" in payload:
            payload = payload["cases"]

        if not isinstance(payload, list):
            raise Stage4CaseLoadError(
                "Few-shot case file must contain a JSON list of case objects "
                "or an object with a 'cases' list."
            )

        cases: list[Stage4FewShotCase] = []
        seen_case_ids: set[str] = set()
        for item in payload:
            if not isinstance(item, dict):
                raise Stage4CaseLoadError(
                    "Each few-shot case must be a JSON object with the required fields."
                )

            case = Stage4FewShotCase.from_mapping(item)
            if case.case_id in seen_case_ids:
                raise Stage4CaseLoadError(f"Duplicate case_id found: {case.case_id}")
            seen_case_ids.add(case.case_id)
            cases.append(case)

        return cases[:limit] if limit is not None else cases


class FewShotPromptBuilder:
    """Build explicit, inspectable prompts for Stage 4 few-shot / zero-shot competence."""

    def build_zero_shot_prompt(self, case: Stage4FewShotCase) -> str:
        return (
            "You are being evaluated on a bounded Stage 4 few-shot / zero-shot competence benchmark.\n"
            "This is the zero-shot condition.\n"
            "You must solve the task from the written task description and instruction only.\n"
            "Do not use any correction loop or ask follow-up questions.\n"
            "Return only the final answer. Do not explain your reasoning.\n\n"
            "=== TASK FAMILY ===\n"
            f"{case.task_family}\n\n"
            "=== TASK DESCRIPTION ===\n"
            f"{case.task_description}\n\n"
            "=== ZERO-SHOT INSTRUCTION ===\n"
            f"{case.zero_shot_instruction}\n\n"
            "=== TARGET INPUT ===\n"
            f"{case.target_input}\n\n"
            "=== FINAL ANSWER ==="
        )

    def build_few_shot_prompt(self, case: Stage4FewShotCase) -> str:
        lines = [
            "You are being evaluated on a bounded Stage 4 few-shot / zero-shot competence benchmark.",
            "This is the few-shot condition.",
            "You must solve the task from the same written task description plus a tiny number of demonstrations.",
            "The demonstrations show the pattern, but they are not corrections to any earlier answer.",
            "Return only the final answer. Do not explain your reasoning.",
            "",
            "=== TASK FAMILY ===",
            case.task_family,
            "",
            "=== TASK DESCRIPTION ===",
            case.task_description,
            "",
            "=== ZERO-SHOT INSTRUCTION ===",
            case.zero_shot_instruction,
            "",
            "=== FEW-SHOT DEMONSTRATIONS ===",
        ]

        for index, example in enumerate(case.few_shot_examples, start=1):
            lines.extend(
                [
                    f"Example {index} input:",
                    example.example_input,
                    f"Example {index} output:",
                    example.example_output,
                    "",
                ]
            )

        lines.extend(
            [
                "=== TARGET INPUT ===",
                case.target_input,
                "",
                "=== FINAL ANSWER ===",
            ]
        )
        return "\n".join(lines)


@dataclass
class FewShotCaseResult:
    """Structured result for one Stage 4 few-shot / zero-shot case."""

    case_id: str
    task_family: str
    scoring_type: str
    expected_answer: str
    zero_shot_answer: str
    few_shot_answer: str
    zero_shot_pass: bool
    few_shot_pass: bool
    few_shot_helped: bool
    regression: bool
    zero_shot_scoring_error: str | None = None
    few_shot_scoring_error: str | None = None
    probable_failure_reason: str | None = None


class FewShotEvaluator:
    """Run zero-shot versus few-shot evaluation for Stage 4."""

    def __init__(
        self,
        ollama_client: OllamaChatClient,
        model: str,
        temperature: float,
        prompt_builder: FewShotPromptBuilder | None = None,
        scoring_engine: ScoringEngine | None = None,
    ) -> None:
        self.ollama_client = ollama_client
        self.model = model
        self.temperature = temperature
        self.prompt_builder = prompt_builder if prompt_builder is not None else FewShotPromptBuilder()
        self.scoring_engine = scoring_engine if scoring_engine is not None else ScoringEngine()

    def evaluate_case(self, case: Stage4FewShotCase) -> FewShotCaseResult:
        zero_shot_answer = self._run_single_prompt(self.prompt_builder.build_zero_shot_prompt(case))
        few_shot_answer = self._run_single_prompt(self.prompt_builder.build_few_shot_prompt(case))

        zero_shot_pass, zero_shot_scoring_error = self.scoring_engine.score_answer(
            answer=zero_shot_answer,
            expected_answer=case.expected_answer,
            scoring_type=case.scoring_type,
        )
        few_shot_pass, few_shot_scoring_error = self.scoring_engine.score_answer(
            answer=few_shot_answer,
            expected_answer=case.expected_answer,
            scoring_type=case.scoring_type,
        )

        few_shot_helped = (not zero_shot_pass) and few_shot_pass
        regression = zero_shot_pass and (not few_shot_pass)
        probable_failure_reason = self._infer_failure_reason(
            zero_shot_answer=zero_shot_answer,
            few_shot_answer=few_shot_answer,
            zero_shot_pass=zero_shot_pass,
            few_shot_pass=few_shot_pass,
            zero_shot_scoring_error=zero_shot_scoring_error,
            few_shot_scoring_error=few_shot_scoring_error,
        )

        return FewShotCaseResult(
            case_id=case.case_id,
            task_family=case.task_family,
            scoring_type=case.scoring_type,
            expected_answer=case.expected_answer,
            zero_shot_answer=zero_shot_answer,
            few_shot_answer=few_shot_answer,
            zero_shot_pass=zero_shot_pass,
            few_shot_pass=few_shot_pass,
            few_shot_helped=few_shot_helped,
            regression=regression,
            zero_shot_scoring_error=zero_shot_scoring_error,
            few_shot_scoring_error=few_shot_scoring_error,
            probable_failure_reason=probable_failure_reason,
        )

    def evaluate_cases(self, cases: Sequence[Stage4FewShotCase]) -> list[FewShotCaseResult]:
        return [self.evaluate_case(case) for case in cases]

    def build_scorecard(self, results: Sequence[FewShotCaseResult]) -> dict[str, Any]:
        total_cases = len(results)
        zero_shot_pass_count = sum(result.zero_shot_pass for result in results)
        few_shot_pass_count = sum(result.few_shot_pass for result in results)
        few_shot_improvement_count = sum(result.few_shot_helped for result in results)
        regression_count = sum(result.regression for result in results)

        zero_shot_pass_rate = (zero_shot_pass_count / total_cases) if total_cases else 0.0
        few_shot_pass_rate = (few_shot_pass_count / total_cases) if total_cases else 0.0

        return {
            "total_cases": total_cases,
            "zero_shot_pass_count": zero_shot_pass_count,
            "few_shot_pass_count": few_shot_pass_count,
            "few_shot_improvement_count": few_shot_improvement_count,
            "regression_count": regression_count,
            "zero_shot_pass_rate": round(zero_shot_pass_rate, 4),
            "pass_rate": round(few_shot_pass_rate, 4),
            "few_shot_pass_rate": round(few_shot_pass_rate, 4),
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
        zero_shot_answer: str,
        few_shot_answer: str,
        zero_shot_pass: bool,
        few_shot_pass: bool,
        zero_shot_scoring_error: str | None,
        few_shot_scoring_error: str | None,
    ) -> str | None:
        if zero_shot_pass and few_shot_pass:
            return None

        if zero_shot_scoring_error or few_shot_scoring_error:
            problems = [item for item in [zero_shot_scoring_error, few_shot_scoring_error] if item]
            return " ; ".join(problems)

        if zero_shot_answer.startswith("ERROR:") or few_shot_answer.startswith("ERROR:"):
            return "A model call failed during evaluation."

        if zero_shot_pass and not few_shot_pass:
            return "Few-shot demonstrations appear to have distracted a previously correct zero-shot answer."

        if (not zero_shot_pass) and (not few_shot_pass):
            zero_norm = self.scoring_engine.normalize_whitespace(zero_shot_answer).casefold()
            few_norm = self.scoring_engine.normalize_whitespace(few_shot_answer).casefold()
            if zero_norm == few_norm:
                return "Few-shot demonstrations did not materially change the answer."
            return "The task description or demonstrations were still not applied correctly."

        return None


class Stage5CaseLoadError(Exception):
    """Raised when the reasoning case file cannot be loaded safely."""


@dataclass(frozen=True)
class Stage5ReasoningCase:
    """One curated Stage 5 robust reasoning case."""

    case_id: str
    task_family: str
    task_text: str
    expected_answer: str
    scoring_type: str
    reasoning_focus: str
    novelty_notes: str
    notes: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "Stage5ReasoningCase":
        required_fields = [
            "case_id",
            "task_family",
            "task_text",
            "expected_answer",
            "scoring_type",
            "reasoning_focus",
            "novelty_notes",
            "notes",
        ]
        missing = [field_name for field_name in required_fields if field_name not in mapping]
        if missing:
            raise Stage5CaseLoadError(
                "Reasoning case is missing required field(s): " + ", ".join(missing)
            )

        values: dict[str, str] = {}
        for field_name in required_fields:
            values[field_name] = require_non_empty_string(
                mapping=mapping,
                field_name=field_name,
                error_type=Stage5CaseLoadError,
                object_label="Reasoning case",
            )

        scoring_key = " ".join(values["scoring_type"].strip().lower().replace("_", " ").split())
        if scoring_key not in ALLOWED_REASONING_SCORING_TYPES:
            allowed = ", ".join(sorted(ALLOWED_REASONING_SCORING_TYPES))
            raise Stage5CaseLoadError(
                f"Unsupported Stage 5 scoring_type '{values['scoring_type']}'. Allowed values: {allowed}."
            )

        return cls(**values)


class ReasoningCaseLoader:
    """Load and validate the Stage 5 reasoning benchmark cases."""

    def load(self, path: str, limit: int | None = None) -> list[Stage5ReasoningCase]:
        case_path = Path(path)
        if not case_path.exists():
            raise Stage5CaseLoadError(f"Reasoning case file was not found: {case_path}")

        try:
            payload = read_json_payload(case_path, "reasoning case")
        except ValueError as exc:
            raise Stage5CaseLoadError(str(exc)) from exc

        if isinstance(payload, dict) and "cases" in payload:
            payload = payload["cases"]

        if not isinstance(payload, list):
            raise Stage5CaseLoadError(
                "Reasoning case file must contain a JSON list of case objects "
                "or an object with a 'cases' list."
            )

        cases: list[Stage5ReasoningCase] = []
        seen_case_ids: set[str] = set()
        for item in payload:
            if not isinstance(item, dict):
                raise Stage5CaseLoadError(
                    "Each reasoning case must be a JSON object with the required fields."
                )

            case = Stage5ReasoningCase.from_mapping(item)
            if case.case_id in seen_case_ids:
                raise Stage5CaseLoadError(f"Duplicate case_id found: {case.case_id}")
            seen_case_ids.add(case.case_id)
            cases.append(case)

        return cases[:limit] if limit is not None else cases


@dataclass(frozen=True)
class ReasoningArtifact:
    """Compact reasoning artifact requested from the model."""

    extracted_facts: list[str]
    transformation_or_rule: str
    tentative_answer: str
    verification_note: str
    final_answer: str


@dataclass(frozen=True)
class VerificationArtifact:
    """Compact verification artifact produced in the bounded verification pass."""

    verification_note: str
    verified_final_answer: str


class ReasoningPromptBuilder:
    """Build explicit direct, structured, and verification prompts for Stage 5."""

    def build_direct_prompt(self, case: Stage5ReasoningCase) -> str:
        return (
            "You are being evaluated on a bounded Stage 5 robust reasoning benchmark.\n"
            "This is the direct-answer baseline condition.\n"
            "Solve the task directly with no structured reasoning scaffold.\n"
            "Return only the final answer. Do not explain your reasoning.\n\n"
            "=== TASK FAMILY ===\n"
            f"{case.task_family}\n\n"
            "=== TASK ===\n"
            f"{case.task_text}\n\n"
            "=== FINAL ANSWER ==="
        )

    def build_reasoning_prompt(self, case: Stage5ReasoningCase) -> str:
        return (
            "You are being evaluated on a bounded Stage 5 robust reasoning benchmark.\n"
            "This is the structured reasoning condition.\n"
            "Return JSON only. Do not use markdown fences.\n"
            "Do not provide hidden chain-of-thought or long essays.\n"
            "Fill this exact schema with brief values:\n"
            "{\n"
            '  "extracted_facts": ["fact 1", "fact 2"],\n'
            '  "transformation_or_rule": "brief rule or operation used",\n'
            '  "tentative_answer": "candidate answer",\n'
            '  "verification_note": "brief self-check note",\n'
            '  "final_answer": "best current final answer"\n'
            "}\n"
            "Requirements:\n"
            "- extracted_facts must be a short list of 1 to 5 brief strings taken from the task.\n"
            "- Keep every field compact and inspectable.\n"
            "- final_answer must contain the answer you currently endorse.\n\n"
            "=== TASK FAMILY ===\n"
            f"{case.task_family}\n\n"
            "=== REASONING FOCUS ===\n"
            f"{case.reasoning_focus}\n\n"
            "=== TASK ===\n"
            f"{case.task_text}"
        )

    def build_verification_prompt(
        self,
        case: Stage5ReasoningCase,
        artifact: ReasoningArtifact,
    ) -> str:
        artifact_json = json.dumps(asdict(artifact), ensure_ascii=False, indent=2)
        return (
            "You are being evaluated on a bounded Stage 5 robust reasoning benchmark.\n"
            "This is the verification condition.\n"
            "Check whether the candidate reasoning artifact is consistent with the task.\n"
            "Return JSON only. Do not use markdown fences.\n"
            "Return this exact schema:\n"
            "{\n"
            '  "verification_note": "brief note about whether the candidate is consistent",\n'
            '  "verified_final_answer": "the final answer to use for scoring"\n'
            "}\n"
            "If the candidate final answer is wrong, correct it.\n"
            "Keep the note brief and use the verified_final_answer field for the answer itself.\n\n"
            "=== TASK FAMILY ===\n"
            f"{case.task_family}\n\n"
            "=== TASK ===\n"
            f"{case.task_text}\n\n"
            "=== CANDIDATE REASONING ARTIFACT ===\n"
            f"{artifact_json}"
        )


class ReasoningOutputParser:
    """Safely parse the structured Stage 5 reasoning outputs."""

    def parse_reasoning_artifact(
        self,
        raw_output: str,
    ) -> tuple[ReasoningArtifact | None, str | None]:
        try:
            payload = self._extract_first_json_object(raw_output)
        except ValueError as exc:
            return None, str(exc)

        if not isinstance(payload, dict):
            return None, "Reasoning output must be a JSON object."

        required_keys = {
            "extracted_facts",
            "transformation_or_rule",
            "tentative_answer",
            "verification_note",
            "final_answer",
        }
        missing = [key for key in required_keys if key not in payload]
        if missing:
            return None, "Reasoning output is missing required key(s): " + ", ".join(sorted(missing))

        extracted_facts = payload.get("extracted_facts")
        if not isinstance(extracted_facts, list) or not extracted_facts:
            return None, "Reasoning output field 'extracted_facts' must be a non-empty JSON list."

        normalized_facts: list[str] = []
        for item in extracted_facts:
            if not isinstance(item, str) or not item.strip():
                return None, "Each extracted fact must be a non-empty string."
            normalized_facts.append(item.strip())

        if len(normalized_facts) > 5:
            return None, "Reasoning output field 'extracted_facts' must contain at most 5 items."

        transformation_or_rule = payload.get("transformation_or_rule")
        tentative_answer = payload.get("tentative_answer")
        verification_note = payload.get("verification_note")
        final_answer = payload.get("final_answer")

        string_fields = {
            "transformation_or_rule": transformation_or_rule,
            "tentative_answer": tentative_answer,
            "verification_note": verification_note,
            "final_answer": final_answer,
        }
        for key, value in string_fields.items():
            if not isinstance(value, str) or not value.strip():
                return None, f"Reasoning output field '{key}' must be a non-empty string."

        return (
            ReasoningArtifact(
                extracted_facts=normalized_facts,
                transformation_or_rule=transformation_or_rule.strip(),
                tentative_answer=tentative_answer.strip(),
                verification_note=verification_note.strip(),
                final_answer=final_answer.strip(),
            ),
            None,
        )

    def parse_verification_output(
        self,
        raw_output: str,
    ) -> tuple[VerificationArtifact | None, str | None]:
        try:
            payload = self._extract_first_json_object(raw_output)
        except ValueError as exc:
            return None, str(exc)

        if not isinstance(payload, dict):
            return None, "Verification output must be a JSON object."

        required_keys = {"verification_note", "verified_final_answer"}
        missing = [key for key in required_keys if key not in payload]
        if missing:
            return None, "Verification output is missing required key(s): " + ", ".join(sorted(missing))

        verification_note = payload.get("verification_note")
        verified_final_answer = payload.get("verified_final_answer")

        if not isinstance(verification_note, str) or not verification_note.strip():
            return None, "Verification output field 'verification_note' must be a non-empty string."
        if not isinstance(verified_final_answer, str) or not verified_final_answer.strip():
            return None, "Verification output field 'verified_final_answer' must be a non-empty string."

        return (
            VerificationArtifact(
                verification_note=verification_note.strip(),
                verified_final_answer=verified_final_answer.strip(),
            ),
            None,
        )

    def _extract_first_json_object(self, raw_output: str) -> Any:
        text = raw_output.strip()
        if not text:
            raise ValueError("Model returned an empty structured output.")

        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                payload, _end = decoder.raw_decode(text[index:])
                return payload
            except json.JSONDecodeError:
                continue

        raise ValueError("Could not find a valid JSON object in the model output.")


@dataclass
class ReasoningCaseResult:
    """Structured result for one Stage 5 reasoning case."""

    case_id: str
    task_family: str
    scoring_type: str
    expected_answer: str
    direct_answer: str
    reasoned_raw_output: str
    parsed_reasoning_artifact: dict[str, Any] | None
    verified_final_answer: str
    direct_pass: bool
    reasoned_pass: bool
    reasoning_helped: bool
    regression: bool
    direct_scoring_error: str | None = None
    reasoned_scoring_error: str | None = None
    parse_error: str | None = None
    probable_failure_reason: str | None = None
    reasoned_answer: str | None = None
    verification_raw_output: str | None = None
    verification_note: str | None = None


class ReasoningEvaluator:
    """Run direct-answer versus structured-and-verified reasoning for Stage 5."""

    def __init__(
        self,
        ollama_client: OllamaChatClient,
        model: str,
        temperature: float,
        prompt_builder: ReasoningPromptBuilder | None = None,
        output_parser: ReasoningOutputParser | None = None,
        scoring_engine: ScoringEngine | None = None,
    ) -> None:
        self.ollama_client = ollama_client
        self.model = model
        self.temperature = temperature
        self.prompt_builder = prompt_builder if prompt_builder is not None else ReasoningPromptBuilder()
        self.output_parser = output_parser if output_parser is not None else ReasoningOutputParser()
        self.scoring_engine = scoring_engine if scoring_engine is not None else ScoringEngine()

    def evaluate_case(self, case: Stage5ReasoningCase) -> ReasoningCaseResult:
        direct_answer = self._run_single_prompt(self.prompt_builder.build_direct_prompt(case))
        reasoned_raw_output = self._run_single_prompt(self.prompt_builder.build_reasoning_prompt(case))

        parsed_artifact, parse_error = self.output_parser.parse_reasoning_artifact(reasoned_raw_output)
        reasoned_answer = parsed_artifact.final_answer if parsed_artifact is not None else None

        verification_raw_output: str | None = None
        verification_note: str | None = None
        verified_final_answer = ""

        if parsed_artifact is not None:
            verification_raw_output = self._run_single_prompt(
                self.prompt_builder.build_verification_prompt(case, parsed_artifact)
            )
            parsed_verification, verification_parse_error = self.output_parser.parse_verification_output(
                verification_raw_output
            )

            if verification_parse_error:
                parse_error = self._combine_errors(parse_error, verification_parse_error)
                verification_note = "Verification output could not be parsed cleanly; using reasoned final answer."
                verified_final_answer = parsed_artifact.final_answer
            else:
                assert parsed_verification is not None
                verification_note = parsed_verification.verification_note
                verified_final_answer = parsed_verification.verified_final_answer
        else:
            verification_note = None
            verified_final_answer = ""

        direct_pass, direct_scoring_error = self.scoring_engine.score_answer(
            answer=direct_answer,
            expected_answer=case.expected_answer,
            scoring_type=case.scoring_type,
        )
        reasoned_pass, reasoned_scoring_error = self.scoring_engine.score_answer(
            answer=verified_final_answer,
            expected_answer=case.expected_answer,
            scoring_type=case.scoring_type,
        )

        reasoning_helped = (not direct_pass) and reasoned_pass
        regression = direct_pass and (not reasoned_pass)

        probable_failure_reason = self._infer_failure_reason(
            direct_answer=direct_answer,
            verified_final_answer=verified_final_answer,
            direct_pass=direct_pass,
            reasoned_pass=reasoned_pass,
            direct_scoring_error=direct_scoring_error,
            reasoned_scoring_error=reasoned_scoring_error,
            parse_error=parse_error,
        )

        return ReasoningCaseResult(
            case_id=case.case_id,
            task_family=case.task_family,
            scoring_type=case.scoring_type,
            expected_answer=case.expected_answer,
            direct_answer=direct_answer,
            reasoned_raw_output=reasoned_raw_output,
            parsed_reasoning_artifact=asdict(parsed_artifact) if parsed_artifact is not None else None,
            verified_final_answer=verified_final_answer,
            direct_pass=direct_pass,
            reasoned_pass=reasoned_pass,
            reasoning_helped=reasoning_helped,
            regression=regression,
            direct_scoring_error=direct_scoring_error,
            reasoned_scoring_error=reasoned_scoring_error,
            parse_error=parse_error,
            probable_failure_reason=probable_failure_reason,
            reasoned_answer=reasoned_answer,
            verification_raw_output=verification_raw_output,
            verification_note=verification_note,
        )

    def evaluate_cases(self, cases: Sequence[Stage5ReasoningCase]) -> list[ReasoningCaseResult]:
        return [self.evaluate_case(case) for case in cases]

    def build_scorecard(self, results: Sequence[ReasoningCaseResult]) -> dict[str, Any]:
        total_cases = len(results)
        direct_pass_count = sum(result.direct_pass for result in results)
        reasoned_pass_count = sum(result.reasoned_pass for result in results)
        reasoning_improvement_count = sum(result.reasoning_helped for result in results)
        regression_count = sum(result.regression for result in results)
        parse_error_count = sum(1 for result in results if result.parse_error)

        direct_pass_rate = (direct_pass_count / total_cases) if total_cases else 0.0
        reasoned_pass_rate = (reasoned_pass_count / total_cases) if total_cases else 0.0

        return {
            "total_cases": total_cases,
            "direct_pass_count": direct_pass_count,
            "reasoned_pass_count": reasoned_pass_count,
            "reasoning_improvement_count": reasoning_improvement_count,
            "regression_count": regression_count,
            "direct_pass_rate": round(direct_pass_rate, 4),
            "reasoned_pass_rate": round(reasoned_pass_rate, 4),
            "parse_error_count": parse_error_count,
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

    def _combine_errors(self, existing: str | None, new_error: str | None) -> str | None:
        if not new_error:
            return existing
        if not existing:
            return new_error
        return f"{existing} ; {new_error}"

    def _infer_failure_reason(
        self,
        direct_answer: str,
        verified_final_answer: str,
        direct_pass: bool,
        reasoned_pass: bool,
        direct_scoring_error: str | None,
        reasoned_scoring_error: str | None,
        parse_error: str | None,
    ) -> str | None:
        if direct_pass and reasoned_pass:
            return None

        if direct_scoring_error or reasoned_scoring_error:
            problems = [item for item in [direct_scoring_error, reasoned_scoring_error] if item]
            return " ; ".join(problems)

        if parse_error:
            return f"Structured reasoning or verification output was malformed: {parse_error}"

        if direct_answer.startswith("ERROR:") or verified_final_answer.startswith("ERROR:"):
            return "A model call failed during evaluation."

        if direct_pass and not reasoned_pass:
            return "The reasoning scaffold or verification pass destabilized a previously correct direct answer."

        if (not direct_pass) and (not reasoned_pass):
            direct_norm = self.scoring_engine.normalize_whitespace(direct_answer).casefold()
            reasoned_norm = self.scoring_engine.normalize_whitespace(verified_final_answer).casefold()
            if direct_norm == reasoned_norm:
                return "The structured reasoning path did not materially improve the final answer."
            return "The structured reasoning path changed the answer, but the verified final answer was still incorrect."

        return None


class Stage6CaseLoadError(Exception):
    """Raised when the commonsense case file cannot be loaded safely."""


ALLOWED_COMMONSENSE_SCENARIO_TYPES = {
    "physical object interaction",
    "containment/support/spillage",
    "heat/cold/wetness",
    "breakage/damage",
    "simple temporal routine",
    "simple location/visibility",
    "simple social appropriateness",
    "simple intention / ordinary human behavior",
    "basic everyday safety consequence",
    "object affordance / tool use expectation",
}


@dataclass(frozen=True)
class Stage6CommonSenseCase:
    """One curated Stage 6 commonsense-understanding case."""

    case_id: str
    scenario_type: str
    scenario_text: str
    question: str
    expected_answer: str
    scoring_type: str
    commonsense_focus: str
    hidden_assumption_notes: str
    novelty_notes: str
    notes: str

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "Stage6CommonSenseCase":
        required_fields = [
            "case_id",
            "scenario_type",
            "scenario_text",
            "question",
            "expected_answer",
            "scoring_type",
            "commonsense_focus",
            "hidden_assumption_notes",
            "novelty_notes",
            "notes",
        ]
        missing = [field_name for field_name in required_fields if field_name not in mapping]
        if missing:
            raise Stage6CaseLoadError(
                "Common-sense case is missing required field(s): " + ", ".join(missing)
            )

        values: dict[str, str] = {}
        for field_name in required_fields:
            values[field_name] = require_non_empty_string(
                mapping=mapping,
                field_name=field_name,
                error_type=Stage6CaseLoadError,
                object_label="Common-sense case",
            )

        scenario_key = values["scenario_type"].strip().lower()
        if scenario_key not in ALLOWED_COMMONSENSE_SCENARIO_TYPES:
            allowed = ", ".join(sorted(ALLOWED_COMMONSENSE_SCENARIO_TYPES))
            raise Stage6CaseLoadError(
                f"Unsupported scenario_type '{values['scenario_type']}'. Allowed values: {allowed}."
            )

        scoring_key = " ".join(values["scoring_type"].strip().lower().replace("_", " ").split())
        if scoring_key not in ALLOWED_REASONING_SCORING_TYPES:
            allowed = ", ".join(sorted(ALLOWED_REASONING_SCORING_TYPES))
            raise Stage6CaseLoadError(
                f"Unsupported Stage 6 scoring_type '{values['scoring_type']}'. Allowed values: {allowed}."
            )

        return cls(**values)


class CommonSenseCaseLoader:
    """Load and validate the Stage 6 commonsense benchmark cases."""

    def load(self, path: str, limit: int | None = None) -> list[Stage6CommonSenseCase]:
        case_path = Path(path)
        if not case_path.exists():
            raise Stage6CaseLoadError(f"Common-sense case file was not found: {case_path}")

        try:
            payload = read_json_payload(case_path, "common-sense case")
        except ValueError as exc:
            raise Stage6CaseLoadError(str(exc)) from exc

        if isinstance(payload, dict) and "cases" in payload:
            payload = payload["cases"]

        if not isinstance(payload, list):
            raise Stage6CaseLoadError(
                "Common-sense case file must contain a JSON list of case objects "
                "or an object with a 'cases' list."
            )

        cases: list[Stage6CommonSenseCase] = []
        seen_case_ids: set[str] = set()
        for item in payload:
            if not isinstance(item, dict):
                raise Stage6CaseLoadError(
                    "Each common-sense case must be a JSON object with the required fields."
                )

            case = Stage6CommonSenseCase.from_mapping(item)
            if case.case_id in seen_case_ids:
                raise Stage6CaseLoadError(f"Duplicate case_id found: {case.case_id}")
            seen_case_ids.add(case.case_id)
            cases.append(case)

        return cases[:limit] if limit is not None else cases


@dataclass(frozen=True)
class CommonSenseArtifact:
    """Compact commonsense artifact requested from the model."""

    stated_facts: list[str]
    implied_commonsense_assumptions: list[str]
    everyday_rule_or_expectation: str
    tentative_answer: str
    final_answer: str


@dataclass(frozen=True)
class CommonSenseVerificationArtifact:
    """Compact verification artifact produced in the bounded Stage 6 verification pass."""

    verification_note: str
    verified_final_answer: str


class CommonSensePromptBuilder:
    """Build explicit direct, structured, and verification prompts for Stage 6."""

    def build_direct_prompt(self, case: Stage6CommonSenseCase) -> str:
        return (
            "You are being evaluated on a bounded Stage 6 common-sense understanding benchmark.\n"
            "This is the direct-answer baseline condition.\n"
            "Answer the everyday scenario question directly, with no structured commonsense scaffold.\n"
            "Return only the final answer. Do not explain your reasoning.\n\n"
            "=== SCENARIO TYPE ===\n"
            f"{case.scenario_type}\n\n"
            "=== SCENARIO ===\n"
            f"{case.scenario_text}\n\n"
            "=== QUESTION ===\n"
            f"{case.question}\n\n"
            "=== FINAL ANSWER ==="
        )

    def build_commonsense_prompt(self, case: Stage6CommonSenseCase) -> str:
        return (
            "You are being evaluated on a bounded Stage 6 common-sense understanding benchmark.\n"
            "This is the commonsense-grounded condition.\n"
            "Return JSON only. Do not use markdown fences.\n"
            "Do not provide hidden chain-of-thought or long essays.\n"
            "Fill this exact schema with brief values:\n"
            "{\n"
            '  "stated_facts": ["fact 1", "fact 2"],\n'
            '  "implied_commonsense_assumptions": ["assumption 1", "assumption 2"],\n'
            '  "everyday_rule_or_expectation": "brief everyday rule or expectation",\n'
            '  "tentative_answer": "candidate answer",\n'
            '  "final_answer": "best current final answer"\n'
            "}\n"
            "Requirements:\n"
            "- stated_facts must contain 1 to 6 short facts explicitly stated in the scenario.\n"
            "- implied_commonsense_assumptions must contain 1 to 5 short ordinary-world assumptions.\n"
            "- Keep every field compact and inspectable.\n"
            "- final_answer must contain the answer you currently endorse.\n\n"
            "=== SCENARIO TYPE ===\n"
            f"{case.scenario_type}\n\n"
            "=== COMMONSENSE FOCUS ===\n"
            f"{case.commonsense_focus}\n\n"
            "=== SCENARIO ===\n"
            f"{case.scenario_text}\n\n"
            "=== QUESTION ===\n"
            f"{case.question}"
        )

    def build_verification_prompt(
        self,
        case: Stage6CommonSenseCase,
        artifact: CommonSenseArtifact,
    ) -> str:
        artifact_json = json.dumps(asdict(artifact), ensure_ascii=False, indent=2)
        return (
            "You are being evaluated on a bounded Stage 6 common-sense understanding benchmark.\n"
            "This is the verification condition.\n"
            "Check whether the implied assumptions and candidate answer are consistent with the scenario.\n"
            "Return JSON only. Do not use markdown fences.\n"
            "Return this exact schema:\n"
            "{\n"
            '  "verification_note": "brief note about whether the assumptions and answer fit the scenario",\n'
            '  "verified_final_answer": "the final answer to use for scoring"\n'
            "}\n"
            "If the candidate final answer is wrong, correct it.\n"
            "Keep the note brief and use the verified_final_answer field for the answer itself.\n\n"
            "=== SCENARIO TYPE ===\n"
            f"{case.scenario_type}\n\n"
            "=== SCENARIO ===\n"
            f"{case.scenario_text}\n\n"
            "=== QUESTION ===\n"
            f"{case.question}\n\n"
            "=== CANDIDATE COMMONSENSE ARTIFACT ===\n"
            f"{artifact_json}"
        )


class CommonSenseOutputParser:
    """Safely parse the structured Stage 6 commonsense outputs."""

    def parse_commonsense_artifact(
        self,
        raw_output: str,
    ) -> tuple[CommonSenseArtifact | None, str | None]:
        try:
            payload = self._extract_first_json_object(raw_output)
        except ValueError as exc:
            return None, str(exc)

        if not isinstance(payload, dict):
            return None, "Common-sense output must be a JSON object."

        required_keys = {
            "stated_facts",
            "implied_commonsense_assumptions",
            "everyday_rule_or_expectation",
            "tentative_answer",
            "final_answer",
        }
        missing = [key for key in required_keys if key not in payload]
        if missing:
            return None, "Common-sense output is missing required key(s): " + ", ".join(sorted(missing))

        stated_facts = payload.get("stated_facts")
        implied_assumptions = payload.get("implied_commonsense_assumptions")
        if not isinstance(stated_facts, list) or not stated_facts:
            return None, "Common-sense output field 'stated_facts' must be a non-empty JSON list."
        if not isinstance(implied_assumptions, list) or not implied_assumptions:
            return None, "Common-sense output field 'implied_commonsense_assumptions' must be a non-empty JSON list."

        normalized_facts: list[str] = []
        for item in stated_facts:
            if not isinstance(item, str) or not item.strip():
                return None, "Each stated fact must be a non-empty string."
            normalized_facts.append(item.strip())

        normalized_assumptions: list[str] = []
        for item in implied_assumptions:
            if not isinstance(item, str) or not item.strip():
                return None, "Each implied common-sense assumption must be a non-empty string."
            normalized_assumptions.append(item.strip())

        if len(normalized_facts) > 6:
            return None, "Common-sense output field 'stated_facts' must contain at most 6 items."
        if len(normalized_assumptions) > 5:
            return None, "Common-sense output field 'implied_commonsense_assumptions' must contain at most 5 items."

        everyday_rule_or_expectation = payload.get("everyday_rule_or_expectation")
        tentative_answer = payload.get("tentative_answer")
        final_answer = payload.get("final_answer")

        string_fields = {
            "everyday_rule_or_expectation": everyday_rule_or_expectation,
            "tentative_answer": tentative_answer,
            "final_answer": final_answer,
        }
        for key, value in string_fields.items():
            if not isinstance(value, str) or not value.strip():
                return None, f"Common-sense output field '{key}' must be a non-empty string."

        return (
            CommonSenseArtifact(
                stated_facts=normalized_facts,
                implied_commonsense_assumptions=normalized_assumptions,
                everyday_rule_or_expectation=everyday_rule_or_expectation.strip(),
                tentative_answer=tentative_answer.strip(),
                final_answer=final_answer.strip(),
            ),
            None,
        )

    def parse_verification_output(
        self,
        raw_output: str,
    ) -> tuple[CommonSenseVerificationArtifact | None, str | None]:
        try:
            payload = self._extract_first_json_object(raw_output)
        except ValueError as exc:
            return None, str(exc)

        if not isinstance(payload, dict):
            return None, "Verification output must be a JSON object."

        required_keys = {"verification_note", "verified_final_answer"}
        missing = [key for key in required_keys if key not in payload]
        if missing:
            return None, "Verification output is missing required key(s): " + ", ".join(sorted(missing))

        verification_note = payload.get("verification_note")
        verified_final_answer = payload.get("verified_final_answer")

        if not isinstance(verification_note, str) or not verification_note.strip():
            return None, "Verification output field 'verification_note' must be a non-empty string."
        if not isinstance(verified_final_answer, str) or not verified_final_answer.strip():
            return None, "Verification output field 'verified_final_answer' must be a non-empty string."

        return (
            CommonSenseVerificationArtifact(
                verification_note=verification_note.strip(),
                verified_final_answer=verified_final_answer.strip(),
            ),
            None,
        )

    def _extract_first_json_object(self, raw_output: str) -> Any:
        text = raw_output.strip()
        if not text:
            raise ValueError("Model returned an empty structured output.")

        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                payload, _end = decoder.raw_decode(text[index:])
                return payload
            except json.JSONDecodeError:
                continue

        raise ValueError("Could not find a valid JSON object in the model output.")


@dataclass
class CommonSenseCaseResult:
    """Structured result for one Stage 6 common-sense case."""

    case_id: str
    scenario_type: str
    scoring_type: str
    expected_answer: str
    direct_answer: str
    commonsense_raw_output: str
    parsed_commonsense_artifact: dict[str, Any] | None
    verified_final_answer: str
    direct_pass: bool
    commonsense_pass: bool
    commonsense_helped: bool
    regression: bool
    direct_scoring_error: str | None = None
    commonsense_scoring_error: str | None = None
    parse_error: str | None = None
    probable_failure_reason: str | None = None
    verification_raw_output: str | None = None
    verification_note: str | None = None
    commonsense_grounded_answer: str | None = None


class CommonSenseEvaluator:
    """Run direct-answer versus structured-and-verified commonsense evaluation for Stage 6."""

    def __init__(
        self,
        ollama_client: OllamaChatClient,
        model: str,
        temperature: float,
        prompt_builder: CommonSensePromptBuilder | None = None,
        output_parser: CommonSenseOutputParser | None = None,
        scoring_engine: ScoringEngine | None = None,
    ) -> None:
        self.ollama_client = ollama_client
        self.model = model
        self.temperature = temperature
        self.prompt_builder = prompt_builder if prompt_builder is not None else CommonSensePromptBuilder()
        self.output_parser = output_parser if output_parser is not None else CommonSenseOutputParser()
        self.scoring_engine = scoring_engine if scoring_engine is not None else ScoringEngine()

    def evaluate_case(self, case: Stage6CommonSenseCase) -> CommonSenseCaseResult:
        direct_answer = self._run_single_prompt(self.prompt_builder.build_direct_prompt(case))
        commonsense_raw_output = self._run_single_prompt(self.prompt_builder.build_commonsense_prompt(case))

        parsed_artifact, parse_error = self.output_parser.parse_commonsense_artifact(commonsense_raw_output)
        commonsense_grounded_answer = parsed_artifact.final_answer if parsed_artifact is not None else None

        verification_raw_output: str | None = None
        verification_note: str | None = None
        verified_final_answer = ""

        if parsed_artifact is not None:
            verification_raw_output = self._run_single_prompt(
                self.prompt_builder.build_verification_prompt(case, parsed_artifact)
            )
            parsed_verification, verification_parse_error = self.output_parser.parse_verification_output(
                verification_raw_output
            )

            if verification_parse_error:
                parse_error = self._combine_errors(parse_error, verification_parse_error)
                verification_note = (
                    "Verification output could not be parsed cleanly; using commonsense-grounded final answer."
                )
                verified_final_answer = parsed_artifact.final_answer
            else:
                assert parsed_verification is not None
                verification_note = parsed_verification.verification_note
                verified_final_answer = parsed_verification.verified_final_answer
        else:
            verification_note = None
            verified_final_answer = ""

        direct_pass, direct_scoring_error = self.scoring_engine.score_answer(
            answer=direct_answer,
            expected_answer=case.expected_answer,
            scoring_type=case.scoring_type,
        )
        commonsense_pass, commonsense_scoring_error = self.scoring_engine.score_answer(
            answer=verified_final_answer,
            expected_answer=case.expected_answer,
            scoring_type=case.scoring_type,
        )

        commonsense_helped = (not direct_pass) and commonsense_pass
        regression = direct_pass and (not commonsense_pass)

        probable_failure_reason = self._infer_failure_reason(
            direct_answer=direct_answer,
            verified_final_answer=verified_final_answer,
            direct_pass=direct_pass,
            commonsense_pass=commonsense_pass,
            direct_scoring_error=direct_scoring_error,
            commonsense_scoring_error=commonsense_scoring_error,
            parse_error=parse_error,
        )

        return CommonSenseCaseResult(
            case_id=case.case_id,
            scenario_type=case.scenario_type,
            scoring_type=case.scoring_type,
            expected_answer=case.expected_answer,
            direct_answer=direct_answer,
            commonsense_raw_output=commonsense_raw_output,
            parsed_commonsense_artifact=asdict(parsed_artifact) if parsed_artifact is not None else None,
            verified_final_answer=verified_final_answer,
            direct_pass=direct_pass,
            commonsense_pass=commonsense_pass,
            commonsense_helped=commonsense_helped,
            regression=regression,
            direct_scoring_error=direct_scoring_error,
            commonsense_scoring_error=commonsense_scoring_error,
            parse_error=parse_error,
            probable_failure_reason=probable_failure_reason,
            verification_raw_output=verification_raw_output,
            verification_note=verification_note,
            commonsense_grounded_answer=commonsense_grounded_answer,
        )

    def evaluate_cases(self, cases: Sequence[Stage6CommonSenseCase]) -> list[CommonSenseCaseResult]:
        return [self.evaluate_case(case) for case in cases]

    def build_scorecard(self, results: Sequence[CommonSenseCaseResult]) -> dict[str, Any]:
        total_cases = len(results)
        direct_pass_count = sum(result.direct_pass for result in results)
        commonsense_pass_count = sum(result.commonsense_pass for result in results)
        commonsense_improvement_count = sum(result.commonsense_helped for result in results)
        regression_count = sum(result.regression for result in results)
        parse_error_count = sum(1 for result in results if result.parse_error)

        direct_pass_rate = (direct_pass_count / total_cases) if total_cases else 0.0
        commonsense_pass_rate = (commonsense_pass_count / total_cases) if total_cases else 0.0

        return {
            "total_cases": total_cases,
            "direct_pass_count": direct_pass_count,
            "commonsense_pass_count": commonsense_pass_count,
            "commonsense_improvement_count": commonsense_improvement_count,
            "regression_count": regression_count,
            "parse_error_count": parse_error_count,
            "direct_pass_rate": round(direct_pass_rate, 4),
            "commonsense_pass_rate": round(commonsense_pass_rate, 4),
            "pass_rate": round(commonsense_pass_rate, 4),
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

    def _combine_errors(self, existing: str | None, new_error: str | None) -> str | None:
        if not new_error:
            return existing
        if not existing:
            return new_error
        return f"{existing} ; {new_error}"

    def _infer_failure_reason(
        self,
        direct_answer: str,
        verified_final_answer: str,
        direct_pass: bool,
        commonsense_pass: bool,
        direct_scoring_error: str | None,
        commonsense_scoring_error: str | None,
        parse_error: str | None,
    ) -> str | None:
        if direct_pass and commonsense_pass:
            return None

        if direct_scoring_error or commonsense_scoring_error:
            problems = [item for item in [direct_scoring_error, commonsense_scoring_error] if item]
            return " ; ".join(problems)

        if parse_error:
            return f"Structured commonsense or verification output was malformed: {parse_error}"

        if direct_answer.startswith("ERROR:") or verified_final_answer.startswith("ERROR:"):
            return "A model call failed during evaluation."

        if direct_pass and not commonsense_pass:
            return "The commonsense scaffold or verification pass destabilized a previously correct direct answer."

        if (not direct_pass) and (not commonsense_pass):
            direct_norm = self.scoring_engine.normalize_whitespace(direct_answer).casefold()
            commonsense_norm = self.scoring_engine.normalize_whitespace(verified_final_answer).casefold()
            if direct_norm == commonsense_norm:
                return "The commonsense path did not materially improve the final answer."
            return "The commonsense path changed the answer, but the verified final answer was still incorrect."

        return None


class ScorecardWriter:
    """Write a structured JSON scorecard to disk."""

    def write(self, path: str | Path, scorecard: Mapping[str, Any]) -> Path:
        output_path = Path(path)
        output_path.write_text(json.dumps(scorecard, indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path


class FailureLogWriter:
    """Write human-readable markdown failure logs for all active stages."""

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
            lines.extend(["All transfer-assisted cases passed in this run.", "", "No failure entries were generated."])
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
            lines.extend(["All adapted follow-up cases passed in this run.", "", "No failure entries were generated."])
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

    def write_fewshot_log(self, path: str | Path, results: Sequence[FewShotCaseResult]) -> Path:
        output_path = Path(path)
        failed_results = [result for result in results if not result.few_shot_pass]

        lines: list[str] = [
            "# Stage 4 Failure Log",
            "",
            f"Total failed few-shot cases: {len(failed_results)}",
            "",
        ]

        if not failed_results:
            lines.extend(["All few-shot cases passed in this run.", "", "No failure entries were generated."])
        else:
            for result in failed_results:
                lines.extend(
                    [
                        f"## {result.case_id}",
                        "",
                        f"- Task family: {result.task_family}",
                        f"- Expected answer: `{result.expected_answer}`",
                        f"- Zero-shot answer: `{result.zero_shot_answer}`",
                        f"- Few-shot answer: `{result.few_shot_answer}`",
                        f"- Probable failure reason: {result.probable_failure_reason or 'Unknown'}",
                        "",
                    ]
                )

        output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return output_path

    def write_reasoning_log(self, path: str | Path, results: Sequence[ReasoningCaseResult]) -> Path:
        output_path = Path(path)
        failed_results = [result for result in results if not result.reasoned_pass]

        lines: list[str] = [
            "# Stage 5 Failure Log",
            "",
            f"Total failed verified-reasoned cases: {len(failed_results)}",
            "",
        ]

        if not failed_results:
            lines.extend(["All verified-reasoned cases passed in this run.", "", "No failure entries were generated."])
        else:
            for result in failed_results:
                lines.extend(
                    [
                        f"## {result.case_id}",
                        "",
                        f"- Task family: {result.task_family}",
                        f"- Expected answer: `{result.expected_answer}`",
                        f"- Direct answer: `{result.direct_answer}`",
                        f"- Verified reasoned final answer: `{result.verified_final_answer}`",
                        f"- Parse error: {result.parse_error or 'None'}",
                        f"- Probable failure reason: {result.probable_failure_reason or 'Unknown'}",
                        "",
                    ]
                )

        output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return output_path


    def write_commonsense_log(self, path: str | Path, results: Sequence[CommonSenseCaseResult]) -> Path:
        output_path = Path(path)
        failed_results = [result for result in results if not result.commonsense_pass]

        lines: list[str] = [
            "# Stage 6 Failure Log",
            "",
            f"Total failed verified-commonsense cases: {len(failed_results)}",
            "",
        ]

        if not failed_results:
            lines.extend(
                [
                    "All verified-commonsense cases passed in this run.",
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
                        f"- Scenario type: {result.scenario_type}",
                        f"- Expected answer: `{result.expected_answer}`",
                        f"- Direct answer: `{result.direct_answer}`",
                        f"- Verified commonsense final answer: `{result.verified_final_answer}`",
                        f"- Parse error: {result.parse_error or 'None'}",
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
        self.transfer_case_loader = transfer_case_loader if transfer_case_loader is not None else TransferCaseLoader()
        self.adaptation_case_loader = adaptation_case_loader if adaptation_case_loader is not None else AdaptationCaseLoader()
        self.scorecard_writer = scorecard_writer if scorecard_writer is not None else ScorecardWriter()
        self.failure_log_writer = failure_log_writer if failure_log_writer is not None else FailureLogWriter()

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
        failure_log_path = self.failure_log_writer.write_adaptation_log(self.config.failure_log_out, results)

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


class Stage4CLIApp(Stage3CLIApp):
    """Top-level controller preserving Stage 1-3 behavior and adding Stage 4 modes."""

    def __init__(
        self,
        config: AppConfig,
        ollama_client: OllamaChatClient,
        transfer_case_loader: TransferCaseLoader | None = None,
        adaptation_case_loader: AdaptationCaseLoader | None = None,
        fewshot_case_loader: FewShotCaseLoader | None = None,
        scorecard_writer: ScorecardWriter | None = None,
        failure_log_writer: FailureLogWriter | None = None,
    ) -> None:
        super().__init__(
            config=config,
            ollama_client=ollama_client,
            transfer_case_loader=transfer_case_loader,
            adaptation_case_loader=adaptation_case_loader,
            scorecard_writer=scorecard_writer,
            failure_log_writer=failure_log_writer,
        )
        self.fewshot_case_loader = fewshot_case_loader if fewshot_case_loader is not None else FewShotCaseLoader()

    def run(self) -> int:
        if self.config.mode in {"fewshot-demo", "fewshot-eval"}:
            return self._run_fewshot_modes()
        return super().run()

    def _run_fewshot_modes(self) -> int:
        try:
            cases = self.fewshot_case_loader.load(self.config.cases_path, limit=self.config.limit)
        except Stage4CaseLoadError as exc:
            print(f"Error: {exc}")
            return 1

        if not cases:
            print("Error: The few-shot case file loaded successfully but contained no cases.")
            return 1

        evaluator = FewShotEvaluator(
            ollama_client=self.ollama_client,
            model=self.config.model,
            temperature=self.config.temperature,
        )

        if self.config.mode == "fewshot-demo":
            return self._run_fewshot_demo(evaluator=evaluator, cases=cases)

        return self._run_fewshot_eval(evaluator=evaluator, cases=cases)

    def _run_fewshot_demo(
        self,
        evaluator: FewShotEvaluator,
        cases: Sequence[Stage4FewShotCase],
    ) -> int:
        print(f"Loaded few-shot cases: {len(cases)}")

        for index, case in enumerate(cases, start=1):
            result = evaluator.evaluate_case(case)
            print("-" * 72)
            print(f"Demo case {index}: {case.case_id}")
            print(f"Task family: {case.task_family}")
            print("Task description:")
            print(case.task_description)
            print(f"Zero-shot answer: {result.zero_shot_answer}")
            print(f"Few-shot answer: {result.few_shot_answer}")
            print(f"Expected answer: {case.expected_answer}")
            print(f"Few-shot helped: {'YES' if result.few_shot_helped else 'NO'}")

        return 0

    def _run_fewshot_eval(
        self,
        evaluator: FewShotEvaluator,
        cases: Sequence[Stage4FewShotCase],
    ) -> int:
        results = evaluator.evaluate_cases(cases)
        scorecard = evaluator.build_scorecard(results)

        scorecard_path = self.scorecard_writer.write(self.config.scorecard_out, scorecard)
        failure_log_path = self.failure_log_writer.write_fewshot_log(self.config.failure_log_out, results)

        print("Few-shot evaluation complete.")
        print(f"Total cases: {scorecard['total_cases']}")
        print(f"Zero-shot passes: {scorecard['zero_shot_pass_count']}")
        print(f"Few-shot passes: {scorecard['few_shot_pass_count']}")
        print(f"Few-shot improvements: {scorecard['few_shot_improvement_count']}")
        print(f"Regressions: {scorecard['regression_count']}")
        print(f"Zero-shot pass rate: {scorecard['zero_shot_pass_rate']:.2%}")
        print(f"Few-shot pass rate: {scorecard['few_shot_pass_rate']:.2%}")
        print(f"Scorecard written to: {scorecard_path}")
        print(f"Failure log written to: {failure_log_path}")

        return 0


class Stage5CLIApp(Stage4CLIApp):
    """Top-level controller preserving Stages 1-4 and adding Stage 5 modes."""

    def __init__(
        self,
        config: AppConfig,
        ollama_client: OllamaChatClient,
        transfer_case_loader: TransferCaseLoader | None = None,
        adaptation_case_loader: AdaptationCaseLoader | None = None,
        fewshot_case_loader: FewShotCaseLoader | None = None,
        reasoning_case_loader: ReasoningCaseLoader | None = None,
        scorecard_writer: ScorecardWriter | None = None,
        failure_log_writer: FailureLogWriter | None = None,
    ) -> None:
        super().__init__(
            config=config,
            ollama_client=ollama_client,
            transfer_case_loader=transfer_case_loader,
            adaptation_case_loader=adaptation_case_loader,
            fewshot_case_loader=fewshot_case_loader,
            scorecard_writer=scorecard_writer,
            failure_log_writer=failure_log_writer,
        )
        self.reasoning_case_loader = reasoning_case_loader if reasoning_case_loader is not None else ReasoningCaseLoader()

    def run(self) -> int:
        if self.config.mode in {"reason-demo", "reason-eval"}:
            return self._run_reason_modes()
        return super().run()

    def _run_reason_modes(self) -> int:
        try:
            cases = self.reasoning_case_loader.load(self.config.cases_path, limit=self.config.limit)
        except Stage5CaseLoadError as exc:
            print(f"Error: {exc}")
            return 1

        if not cases:
            print("Error: The reasoning case file loaded successfully but contained no cases.")
            return 1

        evaluator = ReasoningEvaluator(
            ollama_client=self.ollama_client,
            model=self.config.model,
            temperature=self.config.temperature,
        )

        if self.config.mode == "reason-demo":
            return self._run_reason_demo(evaluator=evaluator, cases=cases)

        return self._run_reason_eval(evaluator=evaluator, cases=cases)

    def _run_reason_demo(
        self,
        evaluator: ReasoningEvaluator,
        cases: Sequence[Stage5ReasoningCase],
    ) -> int:
        print(f"Loaded reasoning cases: {len(cases)}")

        for index, case in enumerate(cases, start=1):
            result = evaluator.evaluate_case(case)
            print("-" * 72)
            print(f"Demo case {index}: {case.case_id}")
            print(f"Task family: {case.task_family}")
            print(f"Direct answer: {result.direct_answer}")
            print(f"Reasoned answer: {result.reasoned_answer or 'PARSE FAILED'}")
            print(f"Verified final answer: {result.verified_final_answer or 'N/A'}")
            print(f"Expected answer: {case.expected_answer}")
            print(f"Reasoning helped: {'YES' if result.reasoning_helped else 'NO'}")

        return 0

    def _run_reason_eval(
        self,
        evaluator: ReasoningEvaluator,
        cases: Sequence[Stage5ReasoningCase],
    ) -> int:
        results = evaluator.evaluate_cases(cases)
        scorecard = evaluator.build_scorecard(results)

        scorecard_path = self.scorecard_writer.write(self.config.scorecard_out, scorecard)
        failure_log_path = self.failure_log_writer.write_reasoning_log(self.config.failure_log_out, results)

        print("Reasoning evaluation complete.")
        print(f"Total cases: {scorecard['total_cases']}")
        print(f"Direct passes: {scorecard['direct_pass_count']}")
        print(f"Verified reasoned passes: {scorecard['reasoned_pass_count']}")
        print(f"Reasoning improvements: {scorecard['reasoning_improvement_count']}")
        print(f"Regressions: {scorecard['regression_count']}")
        print(f"Parse errors: {scorecard['parse_error_count']}")
        print(f"Direct pass rate: {scorecard['direct_pass_rate']:.2%}")
        print(f"Verified reasoned pass rate: {scorecard['reasoned_pass_rate']:.2%}")
        print(f"Scorecard written to: {scorecard_path}")
        print(f"Failure log written to: {failure_log_path}")

        return 0



class Stage6CLIApp(Stage5CLIApp):
    """Top-level controller preserving Stages 1-5 and adding Stage 6 modes."""

    def __init__(
        self,
        config: AppConfig,
        ollama_client: OllamaChatClient,
        transfer_case_loader: TransferCaseLoader | None = None,
        adaptation_case_loader: AdaptationCaseLoader | None = None,
        fewshot_case_loader: FewShotCaseLoader | None = None,
        reasoning_case_loader: ReasoningCaseLoader | None = None,
        commonsense_case_loader: CommonSenseCaseLoader | None = None,
        scorecard_writer: ScorecardWriter | None = None,
        failure_log_writer: FailureLogWriter | None = None,
    ) -> None:
        super().__init__(
            config=config,
            ollama_client=ollama_client,
            transfer_case_loader=transfer_case_loader,
            adaptation_case_loader=adaptation_case_loader,
            fewshot_case_loader=fewshot_case_loader,
            reasoning_case_loader=reasoning_case_loader,
            scorecard_writer=scorecard_writer,
            failure_log_writer=failure_log_writer,
        )
        self.commonsense_case_loader = (
            commonsense_case_loader if commonsense_case_loader is not None else CommonSenseCaseLoader()
        )

    def run(self) -> int:
        if self.config.mode in {"commonsense-demo", "commonsense-eval"}:
            return self._run_commonsense_modes()
        return super().run()

    def _run_commonsense_modes(self) -> int:
        try:
            cases = self.commonsense_case_loader.load(self.config.cases_path, limit=self.config.limit)
        except Stage6CaseLoadError as exc:
            print(f"Error: {exc}")
            return 1

        if not cases:
            print("Error: The common-sense case file loaded successfully but contained no cases.")
            return 1

        evaluator = CommonSenseEvaluator(
            ollama_client=self.ollama_client,
            model=self.config.model,
            temperature=self.config.temperature,
        )

        if self.config.mode == "commonsense-demo":
            return self._run_commonsense_demo(evaluator=evaluator, cases=cases)

        return self._run_commonsense_eval(evaluator=evaluator, cases=cases)

    def _run_commonsense_demo(
        self,
        evaluator: CommonSenseEvaluator,
        cases: Sequence[Stage6CommonSenseCase],
    ) -> int:
        print(f"Loaded common-sense cases: {len(cases)}")

        for index, case in enumerate(cases, start=1):
            result = evaluator.evaluate_case(case)
            print("-" * 72)
            print(f"Demo case {index}: {case.case_id}")
            print(f"Scenario type: {case.scenario_type}")
            print(f"Direct answer: {result.direct_answer}")
            print(f"Common-sense grounded answer: {result.commonsense_grounded_answer or 'PARSE FAILED'}")
            print(f"Verified final answer: {result.verified_final_answer or 'N/A'}")
            print(f"Expected answer: {case.expected_answer}")
            print(f"Common-sense grounding helped: {'YES' if result.commonsense_helped else 'NO'}")

        return 0

    def _run_commonsense_eval(
        self,
        evaluator: CommonSenseEvaluator,
        cases: Sequence[Stage6CommonSenseCase],
    ) -> int:
        results = evaluator.evaluate_cases(cases)
        scorecard = evaluator.build_scorecard(results)

        scorecard_path = self.scorecard_writer.write(self.config.scorecard_out, scorecard)
        failure_log_path = self.failure_log_writer.write_commonsense_log(self.config.failure_log_out, results)

        print("Common-sense evaluation complete.")
        print(f"Total cases: {scorecard['total_cases']}")
        print(f"Direct passes: {scorecard['direct_pass_count']}")
        print(f"Verified commonsense passes: {scorecard['commonsense_pass_count']}")
        print(f"Common-sense improvements: {scorecard['commonsense_improvement_count']}")
        print(f"Regressions: {scorecard['regression_count']}")
        print(f"Parse errors: {scorecard['parse_error_count']}")
        print(f"Direct pass rate: {scorecard['direct_pass_rate']:.2%}")
        print(f"Verified commonsense pass rate: {scorecard['commonsense_pass_rate']:.2%}")
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

    app = Stage6CLIApp(config=config, ollama_client=ollama_client)
    return app.run()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
        raise SystemExit(0)
