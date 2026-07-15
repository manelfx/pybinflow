from typing import Any

from .vis import Content, Node
from archinfo.archerror import ArchError
from loguru import logger


class NodeHead(Content):
    name: str = "head"
    columns: list[str] = ["addr", "name", "attributes"]

    def gen_render(self, node: Node) -> None:
        cfg_node = node.obj
        attributes = []
        if cfg_node.is_simprocedure:
            attributes.append(" SIMP")
        if cfg_node.is_syscall:
            attributes.append(" SYSC")
        if cfg_node.no_ret:
            attributes.append(" NORET")

        node.content[self.name] = {
            "data": [
                {
                    "addr": {
                        "content": "({:#08x})".format(cfg_node.addr),
                    },
                    "name": {"content": cfg_node.name, "style": "B"},
                    "attributes": {"content": " ".join(attributes)},
                }
            ],
            "columns": self.columns,
        }


class NodeAsm(Content):
    name: str = "asm"
    columns: list[str] = ["addr", "mnemonic", "operands"]

    def gen_render(self, node: Node) -> None:
        cfg_node: Any = node.obj

        if type(cfg_node).__name__ in ["CFGNode", "CFGENode"]:
            is_syscall = cfg_node.is_syscall
            is_simprocedure = cfg_node.is_simprocedure
        elif type(cfg_node).__name__ == "CodeLocation":
            is_syscall = False
            is_simprocedure = cfg_node.sim_procedure is not None
        elif type(cfg_node).__name__ == "ProgramVariable":
            is_syscall = False
            is_simprocedure = cfg_node.location.sim_procedure is not None
        elif type(cfg_node).__name__ == "BlockNode":
            is_syscall = False
            is_simprocedure = False
        elif type(cfg_node).__name__ == "HookNode":
            return
        elif type(cfg_node).__name__ == "Function":
            return
        elif type(cfg_node).__name__ == "Block":
            is_syscall = False
            is_simprocedure = False
        else:
            return

        if is_simprocedure or is_syscall:
            return None

        try:
            # FIXME -- pp writes "call <fn>" instead of "call <addr>"
            # print(node.obj.block.pp())
            insns = cfg_node.block.capstone.insns
        except (ArchError, KeyError) as e:
            logger.error(str(e))
            insns = []
        except Exception as e:
            logger.exception(e)
            insns = []

        data = []
        for ins in insns:
            data.append(
                {
                    "addr": {"content": "0x%08x:\t" % ins.address, "align": "LEFT"},
                    "mnemonic": {"content": ins.mnemonic, "align": "LEFT"},
                    "operands": {"content": ins.op_str, "align": "LEFT"},
                    "_ins": ins,
                    "_addr": ins.address,
                }
            )

        node.content[self.name] = {
            "data": data,
            "columns": self.columns,
        }
