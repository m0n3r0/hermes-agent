import importlib
import sys
import types
from contextlib import nullcontext

import pytest


# Keep this test file self-contained: importing cli.py pulls prompt_toolkit and
# many tool modules; these stubs are enough for HermesCLI construction without
# launching the real TUI stack.
def _install_prompt_toolkit_stubs():
    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

    class _Condition:
        def __init__(self, func):
            self.func = func

        def __bool__(self):
            return bool(self.func())

    class _ANSI(str):
        pass

    root = types.ModuleType("prompt_toolkit")
    history = types.ModuleType("prompt_toolkit.history")
    styles = types.ModuleType("prompt_toolkit.styles")
    patch_stdout = types.ModuleType("prompt_toolkit.patch_stdout")
    application = types.ModuleType("prompt_toolkit.application")
    layout = types.ModuleType("prompt_toolkit.layout")
    processors = types.ModuleType("prompt_toolkit.layout.processors")
    filters = types.ModuleType("prompt_toolkit.filters")
    dimension = types.ModuleType("prompt_toolkit.layout.dimension")
    menus = types.ModuleType("prompt_toolkit.layout.menus")
    widgets = types.ModuleType("prompt_toolkit.widgets")
    key_binding = types.ModuleType("prompt_toolkit.key_binding")
    completion = types.ModuleType("prompt_toolkit.completion")
    formatted_text = types.ModuleType("prompt_toolkit.formatted_text")

    history.FileHistory = _Dummy
    styles.Style = _Dummy
    patch_stdout.patch_stdout = lambda *args, **kwargs: nullcontext()
    application.Application = _Dummy
    layout.Layout = _Dummy
    layout.HSplit = _Dummy
    layout.Window = _Dummy
    layout.FormattedTextControl = _Dummy
    layout.ConditionalContainer = _Dummy
    processors.Processor = _Dummy
    processors.Transformation = _Dummy
    processors.PasswordProcessor = _Dummy
    processors.ConditionalProcessor = _Dummy
    filters.Condition = _Condition
    dimension.Dimension = _Dummy
    menus.CompletionsMenu = _Dummy
    widgets.TextArea = _Dummy
    key_binding.KeyBindings = _Dummy
    completion.Completer = _Dummy
    completion.Completion = _Dummy
    formatted_text.ANSI = _ANSI
    root.print_formatted_text = lambda *args, **kwargs: None

    for name, module in {
        "prompt_toolkit": root,
        "prompt_toolkit.history": history,
        "prompt_toolkit.styles": styles,
        "prompt_toolkit.patch_stdout": patch_stdout,
        "prompt_toolkit.application": application,
        "prompt_toolkit.layout": layout,
        "prompt_toolkit.layout.processors": processors,
        "prompt_toolkit.filters": filters,
        "prompt_toolkit.layout.dimension": dimension,
        "prompt_toolkit.layout.menus": menus,
        "prompt_toolkit.widgets": widgets,
        "prompt_toolkit.key_binding": key_binding,
        "prompt_toolkit.completion": completion,
        "prompt_toolkit.formatted_text": formatted_text,
    }.items():
        sys.modules.setdefault(name, module)


@pytest.fixture(autouse=True)
def _restore_cli_modules():
    prefixes = ("tools", "cli", "run_agent")
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if any(name == p or name.startswith(p + ".") for p in prefixes)
    }
    yield
    for name in list(sys.modules):
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            sys.modules.pop(name, None)
    sys.modules.update(original_modules)


def _import_cli():
    for name in list(sys.modules):
        if name == "cli" or name == "run_agent" or name == "tools" or name.startswith("tools."):
            sys.modules.pop(name, None)
    if "firecrawl" not in sys.modules:
        sys.modules["firecrawl"] = types.SimpleNamespace(Firecrawl=object)
    try:
        importlib.import_module("prompt_toolkit")
    except ModuleNotFoundError:
        _install_prompt_toolkit_stubs()
    return importlib.import_module("cli")


def test_copilot_validation_accepts_mythos_alias(monkeypatch):
    from hermes_cli import models

    monkeypatch.setattr(models, "_resolve_copilot_catalog_api_key", lambda: "tok")
    monkeypatch.setattr(models, "_fetch_github_models", lambda api_key: ["claude-opus-4.7", "gpt-5.4"])

    validation = models.validate_requested_model("mythos", "copilot", api_key="tok")
    assert validation["accepted"] is True
    assert validation["recognized"] is True

    dot_alias = models.validate_requested_model("claude-opus-4.7", "copilot", api_key="tok")
    assert dot_alias["accepted"] is True
    assert dot_alias["recognized"] is True

    hyphen_alias = models.validate_requested_model("claude-opus-4-7", "copilot", api_key="tok")
    assert hyphen_alias["accepted"] is True
    assert hyphen_alias["recognized"] is True
    assert models.normalize_copilot_model_id("claude-opus-4-7", api_key="tok") == "claude-opus-4.7"


def test_copilot_validation_rejects_unknown_model(monkeypatch):
    from hermes_cli import models

    monkeypatch.setattr(models, "_resolve_copilot_catalog_api_key", lambda: "tok")
    monkeypatch.setattr(models, "_fetch_github_models", lambda api_key: ["claude-opus-4.7", "gpt-5.4"])

    validation = models.validate_requested_model("definitely-not-a-real-model-xyz", "copilot", api_key="tok")
    assert validation["accepted"] is False
    assert validation["persist"] is False
    assert validation["recognized"] is False
    assert "GitHub Copilot" in validation["message"]


def test_cli_passes_explicit_copilot_model_to_runtime_resolver(monkeypatch):
    """`hermes chat --provider copilot --model <x>` must resolve Copilot
    routing/API mode against <x>, not the persisted config default.

    This is the guard that keeps Mythos/preview probes honest: without
    target_model propagation, an explicit preview-model probe can silently use
    API-mode decisions from whatever default model was saved last.
    """
    cli = _import_cli()
    calls = []

    def _runtime_resolve(**kwargs):
        calls.append(dict(kwargs))
        return {
            "provider": "copilot",
            "api_mode": "anthropic_messages",
            "base_url": "https://api.githubcopilot.com",
            "api_key": "copilot-runtime-key",
            "source": "test",
        }

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _runtime_resolve)
    monkeypatch.setattr("hermes_cli.runtime_provider.format_runtime_provider_error", lambda exc: str(exc))
    monkeypatch.setattr("hermes_cli.models.normalize_copilot_model_id", lambda model, api_key=None: model)
    monkeypatch.setattr("hermes_cli.models.copilot_model_api_mode", lambda model, api_key=None: "anthropic_messages")
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *args, **kwargs: {"accepted": True, "persist": True, "recognized": True, "message": None},
    )

    shell = cli.HermesCLI(
        provider="copilot",
        model="claude-mythos",
        compact=True,
        max_turns=1,
    )

    assert shell._ensure_runtime_credentials() is True
    assert calls
    assert calls[0]["requested"] == "copilot"
    assert calls[0]["target_model"] == "claude-mythos"
    assert shell.provider == "copilot"
    assert shell.model == "claude-mythos"
    assert shell.api_mode == "anthropic_messages"




def test_cli_passes_fallback_model_to_runtime_resolver(monkeypatch):
    cli = _import_cli()
    from hermes_cli.auth import AuthError

    calls = []

    def _runtime_resolve(**kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise AuthError("primary unavailable")
        return {
            "provider": "copilot",
            "api_mode": "anthropic_messages",
            "base_url": "https://api.githubcopilot.com",
            "api_key": "fallback-runtime-key",
            "source": "test",
        }

    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", _runtime_resolve)
    monkeypatch.setattr("hermes_cli.runtime_provider.format_runtime_provider_error", lambda exc: str(exc))
    monkeypatch.setattr("hermes_cli.models.normalize_copilot_model_id", lambda model, api_key=None: model)
    monkeypatch.setattr("hermes_cli.models.copilot_model_api_mode", lambda model, api_key=None: "anthropic_messages")
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *args, **kwargs: {"accepted": True, "persist": True, "recognized": True, "message": None},
    )

    shell = cli.HermesCLI(
        provider="openai",
        model="gpt-4o",
        compact=True,
        max_turns=1,
    )
    shell._fallback_model = [{"provider": "copilot", "model": "claude-mythos"}]

    assert shell._ensure_runtime_credentials() is True
    assert calls[0]["requested"] == "openai"
    assert calls[0]["target_model"] == "gpt-4o"
    assert calls[1]["requested"] == "copilot"
    assert calls[1]["target_model"] == "claude-mythos"
    assert shell.provider == "copilot"
    assert shell.model == "claude-mythos"
    assert shell.api_mode == "anthropic_messages"


def test_cli_rejects_explicit_unknown_copilot_model(monkeypatch):
    cli = _import_cli()

    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {
            "provider": "copilot",
            "api_mode": "chat_completions",
            "base_url": "https://api.githubcopilot.com",
            "api_key": "copilot-runtime-key",
            "source": "test",
        },
    )
    monkeypatch.setattr("hermes_cli.models.normalize_copilot_model_id", lambda model, api_key=None: model)
    monkeypatch.setattr("hermes_cli.models.copilot_model_api_mode", lambda model, api_key=None: "chat_completions")
    monkeypatch.setattr(
        "hermes_cli.models.validate_requested_model",
        lambda *args, **kwargs: {
            "accepted": False,
            "persist": False,
            "recognized": False,
            "message": "Model `bogus` was not found in the GitHub Copilot model listing.",
        },
    )

    shell = cli.HermesCLI(
        provider="copilot",
        model="bogus",
        compact=True,
        max_turns=1,
    )

    assert shell._ensure_runtime_credentials() is False
    assert shell.agent is None


def test_copilot_model_api_mode_treats_mythos_alias_as_claude_messages():
    from hermes_cli.models import copilot_model_api_mode

    assert copilot_model_api_mode("mythos") == "anthropic_messages"
    assert copilot_model_api_mode("claude-mythos") == "anthropic_messages"


def test_copilot_runtime_api_mode_honors_target_model(monkeypatch):
    import hermes_cli.runtime_provider as rp

    class Entry:
        runtime_api_key = "copilot-api-key"
        runtime_base_url = "https://api.githubcopilot.com"
        source = "test"

    seen = []

    def _mode(model, api_key=None):
        seen.append((model, api_key))
        if model == "claude-mythos":
            return "anthropic_messages"
        return "chat_completions"

    monkeypatch.setattr("hermes_cli.models.copilot_model_api_mode", _mode)

    runtime = rp._resolve_runtime_from_pool_entry(
        provider="copilot",
        entry=Entry(),
        requested_provider="copilot",
        model_cfg={"provider": "copilot", "default": "gpt-5.4"},
        target_model="claude-mythos",
    )

    assert runtime["api_mode"] == "anthropic_messages"
    assert seen == [("claude-mythos", "copilot-api-key")]
