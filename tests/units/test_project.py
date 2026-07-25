"""Fast tests for CFGFast preconditions applied by the project layer."""

from types import SimpleNamespace

from capstone import CS_GRP_CALL, CS_OP_IMM

from bingraph.core import project as project_module


def _terminal_call(addr: int, target: int) -> SimpleNamespace:
    """Create the Capstone instruction shape for a direct five-byte call."""

    return SimpleNamespace(
        address=addr,
        size=5,
        groups=(CS_GRP_CALL,),
        operands=(SimpleNamespace(type=CS_OP_IMM, imm=target),),
    )


def _project(memory_load) -> SimpleNamespace:
    """Create the minimum loader shape needed by the terminal-call check."""

    return SimpleNamespace(
        loader=SimpleNamespace(memory=SimpleNamespace(load=memory_load))
    )


def test_terminal_unmapped_direct_call_is_predeclared_nonreturning(
    monkeypatch,
) -> None:
    """Avoid a fake return when the final direct call cannot return in-image."""

    function_addr = 0x1000
    function_size = 0x20
    callee_addr = 0x2000
    monkeypatch.setattr(
        project_module,
        "decode_raw_capstone_insns",
        lambda *_args: (_terminal_call(0x101B, callee_addr),),
    )
    project = _project(lambda _addr, _size: (_ for _ in ()).throw(KeyError()))

    assert (
        project_module._terminal_unmapped_return_callee(
            project,
            function_addr,
            function_size,
            {callee_addr},
        )
        == callee_addr
    )


def test_terminal_call_with_mapped_return_keeps_normal_inference(monkeypatch) -> None:
    """Do not override CFGFast when the function has a readable continuation."""

    function_addr = 0x1000
    function_size = 0x20
    callee_addr = 0x2000
    monkeypatch.setattr(
        project_module,
        "decode_raw_capstone_insns",
        lambda *_args: (_terminal_call(0x101B, callee_addr),),
    )
    project = _project(lambda _addr, _size: b"\x00")

    assert (
        project_module._terminal_unmapped_return_callee(
            project,
            function_addr,
            function_size,
            {callee_addr},
        )
        is None
    )


def test_terminal_call_to_unknown_target_keeps_normal_inference(monkeypatch) -> None:
    """Do not predeclare unknown direct targets as non-returning functions."""

    function_addr = 0x1000
    function_size = 0x20
    callee_addr = 0x2000
    monkeypatch.setattr(
        project_module,
        "decode_raw_capstone_insns",
        lambda *_args: (_terminal_call(0x101B, callee_addr),),
    )
    project = _project(lambda _addr, _size: (_ for _ in ()).throw(KeyError()))

    assert (
        project_module._terminal_unmapped_return_callee(
            project,
            function_addr,
            function_size,
            set(),
        )
        is None
    )
