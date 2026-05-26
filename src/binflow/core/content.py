from typing import List
from .vis import Content, Node


class NodeHead(Content):
    name: str = "head"
    columns: List[str] = ['addr', 'name', 'attributes']

    def gen_render(self, n: Node) -> None:
        node = n.obj
        attributes=[]
        if node.is_simprocedure:
            attributes.append("SIMP")
        if node.is_syscall:
            attributes.append("SYSC")
        if node.no_ret:
            attributes.append("NORET")

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
    columns: List[str] = ['addr', 'mnemonic', 'operands']

    def gen_render(self, n: Node) -> None:
        node = n.obj

        if type(node).__name__ in ['CFGNode', 'CFGNodeA', 'CFGENode']:
            is_syscall = node.is_syscall
            is_simprocedure = node.is_simprocedure
            addr = node.addr
            size = None
            max_size = node.size
        elif type(node).__name__ == 'CodeLocation':
            is_syscall = False
            is_simprocedure = node.sim_procedure is not None
            addr = node.ins_addr
            size = 1
            max_size = None
        elif type(node).__name__ == 'ProgramVariable':
            is_syscall = False
            is_simprocedure = node.location.sim_procedure is not None
            addr = node.location.ins_addr
            max_size = None
            size = 1
        elif type(node).__name__ == 'BlockNode':
            is_syscall = False
            is_simprocedure = False
            addr = node.addr
            max_size = node.size
            size = None
        elif type(node).__name__ == 'HookNode':
            return
        elif type(node).__name__ == 'Function':
            return
        elif type(node).__name__ == 'Block':
            addr = node.addr
            max_size = None
            size = None
            is_syscall = False
            is_simprocedure = False
        else:
            return

        if is_simprocedure or is_syscall:
            return None

        try:
            insns = n.cfg.project.factory.block(addr=addr, size=max_size, num_inst=size).capstone.insns
        except Exception as e:
            print(e)
            #TODO add logging
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