"""
Abstract base class for all agents providing common functionality: LLM interaction, error handling, retries, timeout management, and metrics tracking.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypeVar, Generic
from datetime import datetime
import threading
import time
import json
import traceback

T = TypeVar('T')


class AgentError(Exception):
    pass


class AgentTimeoutError(AgentError):
    pass


class AgentValidationError(AgentError):
    pass


class AgentExecutionError(AgentError):
    pass


class BaseAgent(ABC, Generic[T]):

    def __init__(
        self,
        name: str,
        model: str,
        client: Any,
        timeout: int = 10,
        max_retries: int = 2,
        retry_delay: float = 0.5,
        temperature: float = 0.1,
        max_tokens: int = 1000,
        verbose: bool = False
    ):
        self.name = name
        self.model = model
        self.client = client
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose

        self._metrics_lock = threading.Lock()
        self.metrics = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "total_duration": 0.0,
            "average_duration": 0.0,
            "retry_count": 0
        }

    def execute(self, input_data: Dict[str, Any]) -> T:
        with self._metrics_lock:
            self.metrics["total_calls"] += 1
        start_time = time.time()

        try:
            self._log(f"Validating input for {self.name}")
            self._validate_input(input_data)

            last_error = None
            for attempt in range(self.max_retries + 1):
                try:
                    if attempt > 0:
                        self._log(f"Retry attempt {attempt}/{self.max_retries}")
                        with self._metrics_lock:
                            self.metrics["retry_count"] += 1
                        time.sleep(min(self.retry_delay * (2 ** (attempt - 1)), 5.0))

                    self._log(f"Executing {self.name}...")
                    raw_output = self._execute_internal(input_data)

                    self._log(f"Formatting output for {self.name}")
                    formatted_output = self._format_output(raw_output)

                    duration = time.time() - start_time
                    self._update_metrics(duration, success=True)
                    self._log(f"{self.name} completed in {duration:.2f}s")

                    return formatted_output

                except (AgentTimeoutError, Exception) as e:
                    last_error = e
                    self._log(f"Attempt {attempt + 1} failed: {str(e)}")

                    if attempt == self.max_retries:
                        break

                    # Rate limit errors need much longer backoff
                    err_msg = str(e).lower()
                    if any(k in err_msg for k in ("429", "rate limit", "rate_limit", "too many requests", "quota")):
                        rate_wait = 15.0 * (2 ** (attempt))  # 30s, 60s, 120s
                        self._log(f"Rate limited — waiting {rate_wait:.0f}s before retry")
                        time.sleep(rate_wait)
                    continue

            duration = time.time() - start_time
            self._update_metrics(duration, success=False)

            error_msg = f"{self.name} failed after {self.max_retries + 1} attempts"
            if last_error:
                error_msg += f": {str(last_error)}"

            raise AgentExecutionError(error_msg)

        except AgentValidationError as e:
            duration = time.time() - start_time
            self._update_metrics(duration, success=False)
            self._log(f"Validation error in {self.name}: {str(e)}")
            raise

        except AgentTimeoutError as e:
            duration = time.time() - start_time
            self._update_metrics(duration, success=False)
            self._log(f"Timeout in {self.name}: {str(e)}")
            raise

        except Exception as e:
            duration = time.time() - start_time
            self._update_metrics(duration, success=False)
            self._log(f"Unexpected error in {self.name}: {str(e)}")
            if self.verbose:
                self._log(f"Traceback: {traceback.format_exc()}")
            raise AgentExecutionError(f"{self.name} failed: {str(e)}")

    @abstractmethod
    def _execute_internal(self, input_data: Dict[str, Any]) -> Any:
        pass

    @abstractmethod
    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def _format_output(self, raw_output: Any) -> T:
        pass

    def _call_llm(
        self,
        messages: list[Dict[str, str]],
        response_format: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        start_time = time.time()

        try:
            temp = temperature if temperature is not None else self.temperature
            tokens = max_tokens if max_tokens is not None else self.max_tokens

            self._log(f"Calling {self.model} (temp={temp}, max_tokens={tokens}, timeout={self.timeout}s)")

            request_params = {
                "model": self.model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": tokens,
            }

            if response_format:
                request_params["response_format"] = response_format

            response = self._run_with_timeout(request_params)

            elapsed = time.time() - start_time
            content = response.choices[0].message.content

            self._log(f"DEBUG: API returned content type = {type(content)}")
            self._log(f"DEBUG: API returned content (first 500): {str(content)[:500] if content else 'None'}")

            self._log(f"LLM responded in {elapsed:.2f}s")

            return content

        except AgentTimeoutError:
            raise

        except Exception as e:
            elapsed = time.time() - start_time
            self._log(f"LLM call failed after {elapsed:.2f}s: {str(e)}")
            raise

    def _run_with_timeout(self, request_params: Dict[str, Any]):
        """Run the LLM API call with timeout enforced via the HTTP client."""
        params = {**request_params, "timeout": self.timeout}
        try:
            return self.client.chat.completions.create(**params)
        except Exception as api_error:
            # Retry without response_format if the server doesn't support it
            if params.get("response_format") and "response_format" in str(api_error):
                self._log("response_format not supported, retrying without it...")
                params.pop("response_format")
                try:
                    return self.client.chat.completions.create(**params)
                except Exception as retry_error:
                    if "Timeout" in type(retry_error).__name__:
                        raise AgentTimeoutError(
                            f"LLM call timed out after {self.timeout}s"
                        )
                    raise
            if "Timeout" in type(api_error).__name__:
                raise AgentTimeoutError(
                    f"LLM call timed out after {self.timeout}s"
                )
            raise

    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        import re

        try:
            # Strip thinking tags from Qwen3 and similar models
            content = re.sub(r'<think>.*?</think>\s*', '', content, flags=re.DOTALL)

            self._log(f"DEBUG: Raw response (first 500 chars): {content[:500]}")

            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]

            if content.endswith("```"):
                content = content[:-3]

            content = content.strip()

            try:
                parsed = json.loads(content)
                return parsed
            except json.JSONDecodeError:
                json_match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    self._log(f"DEBUG: Extracted JSON: {json_str[:200]}")
                    parsed = json.loads(json_str)
                    return parsed
                else:
                    raise

        except json.JSONDecodeError as e:
            self._log(f"DEBUG: Failed to parse. Error: {str(e)}")
            self._log(f"DEBUG: Content type: {type(content)}")
            self._log(f"DEBUG: Content repr: {repr(content[:200])}")
            raise AgentValidationError(
                f"Failed to parse JSON response: {str(e)}\nContent: {content[:200]}"
            )

    def _update_metrics(self, duration: float, success: bool) -> None:
        with self._metrics_lock:
            if success:
                self.metrics["successful_calls"] += 1
            else:
                self.metrics["failed_calls"] += 1

            self.metrics["total_duration"] += duration
            total = self.metrics["total_calls"]
            if total > 0:
                self.metrics["average_duration"] = self.metrics["total_duration"] / total

    def _log(self, message: str) -> None:
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")

    def get_metrics(self) -> Dict[str, Any]:
        with self._metrics_lock:
            return {
                "agent_name": self.name,
                "model": self.model,
                **self.metrics
            }

    def reset_metrics(self) -> None:
        with self._metrics_lock:
            self.metrics = {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "total_duration": 0.0,
                "average_duration": 0.0,
                "retry_count": 0
            }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"model='{self.model}', "
            f"timeout={self.timeout}s"
            f")"
        )
