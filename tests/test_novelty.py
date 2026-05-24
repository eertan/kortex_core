from kortex.sandbox.novelty import (
    NoveltyBranch,
    NoveltyCommandResult,
    NoveltyRequest,
    NoveltyResult,
)


class FakeNoveltyRunner:
    """Command runner test double for novelty branch tests."""

    def __init__(self, results: list[NoveltyCommandResult]) -> None:
        """Store preconfigured command results."""
        self.results = results
        self.calls: list[tuple[list[str], str]] = []

    def run(self, command: list[str], cwd: str) -> NoveltyCommandResult:
        """Capture the command and return the next configured result."""
        self.calls.append((command, cwd))
        return self.results.pop(0)


class FakeNoveltyAgent:
    """Provider-neutral novelty backend test double."""

    provider_name = "fake"

    def __init__(self) -> None:
        """Initialize captured requests."""
        self.requests: list[NoveltyRequest] = []

    def resolve(self, request: NoveltyRequest) -> NoveltyResult:
        """Capture the request and report success."""
        self.requests.append(request)
        return NoveltyResult(
            resolved=True,
            provider=self.provider_name,
            message="fake resolved",
            changed_files=[request.domain_manifest_path],
        )


def test_novelty_branch_delegates_to_provider_neutral_agent(tmp_path):
    fake_agent = FakeNoveltyAgent()
    novelty = NoveltyBranch(
        workspace_path=str(tmp_path),
        novelty_agent=fake_agent,
        validation_command=["pytest", "-q"],
    )

    result = novelty.resolve_impasse_result(
        failed_goal={"fluent": "audit_completed", "args": ["prod_db"]},
        current_state={"server_reachable(prod_db)": True},
        available_actions=["check_disk_space", "clear_cache"],
    )

    assert result.resolved is True
    assert result.provider == "fake"
    assert result.changed_files == [str(tmp_path / "domain.yaml")]
    assert len(fake_agent.requests) == 1
    request = fake_agent.requests[0]
    assert request.validation_command == ["pytest", "-q"]
    assert request.plugins_dir == str(tmp_path / "kortex" / "plugins")


def test_default_pi_backend_dry_run_builds_constrained_prompt(tmp_path):
    novelty = NoveltyBranch(workspace_path=str(tmp_path))

    result = novelty.resolve_impasse_result(
        failed_goal={"fluent": "audit_completed", "args": ["prod_db"]},
        current_state={"server_reachable(prod_db)": True},
        available_actions=["check_disk_space", "clear_cache"],
    )

    assert result.resolved is True
    assert result.provider == "pi"
    assert result.command is not None
    assert result.command[:3] == ["pi", "run", "worker"]
    assert result.prompt is not None
    assert "htn_methods" in result.prompt
    assert "Only if a primitive physical capability is missing" in result.prompt


def test_pi_backend_executes_provider_and_validation_commands(tmp_path):
    runner = FakeNoveltyRunner(
        [
            NoveltyCommandResult(returncode=0, stdout="pi ok"),
            NoveltyCommandResult(returncode=0, stdout="tests ok"),
        ]
    )
    novelty = NoveltyBranch(
        workspace_path=str(tmp_path),
        command_runner=runner,
        execute_pi=True,
        validation_command=["pytest", "-q", "tests/test_generated.py"],
    )

    result = novelty.resolve_impasse_result(
        failed_goal={"fluent": "audit_completed", "args": ["prod_db"]},
        current_state={},
        available_actions=["check_disk_space"],
    )

    assert result.resolved is True
    assert result.provider == "pi"
    assert result.validation_passed is True
    assert len(runner.calls) == 2
    assert runner.calls[0][0][:3] == ["pi", "run", "worker"]
    assert runner.calls[1][0] == ["pytest", "-q", "tests/test_generated.py"]


def test_pi_backend_fails_when_validation_fails(tmp_path):
    runner = FakeNoveltyRunner(
        [
            NoveltyCommandResult(returncode=0, stdout="pi ok"),
            NoveltyCommandResult(returncode=1, stderr="test failed"),
        ]
    )
    novelty = NoveltyBranch(
        workspace_path=str(tmp_path),
        command_runner=runner,
        execute_pi=True,
        validation_command=["pytest", "-q"],
    )

    result = novelty.resolve_impasse_result(
        failed_goal={"fluent": "audit_completed", "args": ["prod_db"]},
        current_state={},
        available_actions=["check_disk_space"],
    )

    assert result.resolved is False
    assert result.validation_passed is False
    assert "validation failed" in result.message.lower()
