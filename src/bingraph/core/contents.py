from .vis import Content, Node
from loguru import logger



class NodeHead(Content):
    name: str = "head"
    columns: list[str] = ['addr', 'name', 'attributes']

    def gen_render(self, n: Node) -> None:
        node = n.obj
        attributes=[]
        if node.is_simprocedure:
            attributes.append(" SIMP")
        if node.is_syscall:
            attributes.append(" SYSC")
        if node.no_ret:
            attributes.append(" NORET")

        n.content[self.name] = {
            'data': [{
                'addr': {
                    'content': "({:#08x})".format(node.addr),
                },
                'name': {
                    'content': node.name,
                    'style':'B'
                },
                'attributes': {
                    'content': ' '.join(attributes)
                }
            }],
            'columns': self.columns
        }


class NodeAsm(Content):
    name: str = "asm"
    columns: list[str] = ['addr', 'mnemonic', 'operands']

    def gen_render(self, n: Node) -> None:
        node = n.obj

        if type(node).__name__ in ['CFGNode', 'CFGENode']:
            is_syscall = node.is_syscall
            is_simprocedure = node.is_simprocedure
        elif type(node).__name__ == 'CodeLocation':
            is_syscall = False
            is_simprocedure = node.sim_procedure is not None
        elif type(node).__name__ == 'ProgramVariable':
            is_syscall = False
            is_simprocedure = node.location.sim_procedure is not None
        elif type(node).__name__ == 'BlockNode':
            is_syscall = False
            is_simprocedure = False
        elif type(node).__name__ == 'HookNode':
            return
        elif type(node).__name__ == 'Function':
            return
        elif type(node).__name__ == 'Block':
            is_syscall = False
            is_simprocedure = False
        else:
            return

        if is_simprocedure or is_syscall:
            return None

        try:
            # FIXME -- pp writes "call <fn>" instead of "call <addr>"
            #print(n.obj.block.pp())
            insns = n.obj.block.capstone.insns
        except Exception as e:
            logger.exception(e)
            insns = []

        data = []
        for ins in insns:
            data.append({
                'addr': {
                    'content': "0x%08x:\t" % ins.address,
                    'align': 'LEFT'
                },
                'mnemonic': {
                    'content': ins.mnemonic,
                    'align': 'LEFT'
                },
                'operands': {
                    'content': ins.op_str,
                    'align': 'LEFT'
                },
                '_ins': ins,
                '_addr': ins.address
            })

        n.content[self.name] = {
            'data': data,
            'columns': self.columns,
        }