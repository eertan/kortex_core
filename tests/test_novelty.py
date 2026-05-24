from kortex.sandbox.novelty import NoveltyBranch, NoveltyCommandResult


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


def test_novelty_branch_dry_run_builds_constrained_pi_prompt(tmp_path):
    novelty = NoveltyBranch(workspace_path=str(tmp_path))

    resolved = novelty.resolve_impasse(
        failed_goal={"fluent": "audit_completed", "args": ["prod_db"]},
        current_state={"server_reachable(prod_db)": True},
        available_actions=["check_disk_space", "clear_cache"],
    )

    assert resolved is True
    assert novelty.last_command is not None
    assert novelty.last_command[:3] == ["pi", "run", "worker"]
    assert novelty.last_prompt is not None
    assert "htn_methods" in novelty.last_prompt
    assert "Only if a primitive physical capability is missing" in novelty.last_prompt


def test_novelty_branch_executes_pi_and_validation_commands(tmp_path):
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

    resolved = novelty.resolve_impasse(
        failed_goal={"fluent": "audit_completed", "args": ["prod_db"]},
        current_state={},
        available_actions=["check_disk_space"],
    )

    assert resolved is True
    assert len(runner.calls) == 2
    assert runner.calls[0][0][:3] == ["pi", "run", "worker"]
    assert runner.calls[1][0] == ["pytest", "-q", "tests/test_generated.py"]


def test_novelty_branch_fails_when_validation_fails(tmp_path):
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

    resolved = novelty.resolve_impasse(
        failed_goal={"fluent": "audit_completed", "args": ["prod_db"]},
        current_state={},
        available_actions=["check_disk_space"],
    )

    assert resolved is False
