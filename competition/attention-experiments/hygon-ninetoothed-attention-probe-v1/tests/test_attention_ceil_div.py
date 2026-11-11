"""Exercise the real DSL loop-bound AST under the pinned backend's signed division semantics."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

PACKAGE=Path(__file__).resolve().parents[1]


def cdiv_signed(a,b):
    magnitude=abs(a)//abs(b)
    return -magnitude if (a<0)!=(b<0) else magnitude


def evaluate_bound(source,n,block):
    tree=ast.parse(source)
    application=next(node for node in tree.body if isinstance(node,ast.FunctionDef)
                     and node.name=='application_without_kv_cache')
    loop=next(node for node in ast.walk(application) if isinstance(node,ast.For)
              and isinstance(node.target,ast.Name) and node.target.id=='j')
    class TensorDivision(ast.NodeTransformer):
        def visit_BinOp(self,node):
            self.generic_visit(node)
            if isinstance(node.op,ast.FloorDiv):
                return ast.copy_location(ast.Call(func=ast.Name(id='cdiv_signed',ctx=ast.Load()),
                                                  args=[node.left,node.right],keywords=[]),node)
            return node
    expression=TensorDivision().visit(ast.Expression(body=loop.iter.args[0]))
    ast.fix_missing_locations(expression)
    return eval(compile(expression,'<DSL-loop-bound>','eval'),{'__builtins__':{}},
                dict(actual_kv_len=n,key=SimpleNamespace(dtype=SimpleNamespace(shape=(block,64))),cdiv_signed=cdiv_signed))


class AttentionCeilDivisionTests(unittest.TestCase):
    def setUp(self):
        spec=importlib.util.spec_from_file_location('attention_ceil_contract',PACKAGE/'attention_contract.py')
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        self.original=(PACKAGE/'upstream_attention.py').read_bytes()

    def test_original_reproduces_zero_single_key_and_drops_tail_block(self):
        self.assertEqual(evaluate_bound(self.original.decode(),1,32),0)
        self.assertEqual(evaluate_bound(self.original.decode(),33,32),1)
        self.assertEqual(evaluate_bound(self.original.decode(),32,32),1)

    def test_candidate_executes_all_blocks_for_short_aligned_and_tail_lengths(self):
        self.assertTrue(hasattr(self.module,'candidate_attention_source'),'positive ceiling-division repair missing')
        candidate=self.module.candidate_attention_source(self.original).decode()
        for block in (16,32,64,128):
            for n in (1,2,17,31,32,33,63,64,65,127,128,129,257,1025):
                with self.subTest(n=n,block=block):
                    self.assertEqual(evaluate_bound(candidate,n,block),len(range(0,n,block)))

    def test_candidate_only_changes_loop_count_and_preserves_original(self):
        self.assertTrue(hasattr(self.module,'candidate_attention_source'),'positive ceiling-division repair missing')
        before=self.original
        candidate=self.module.candidate_attention_source(before)
        self.assertEqual(candidate,before.replace(self.module.CEIL_OLD.encode(),self.module.CEIL_NEW.encode(),1))
        self.assertEqual((PACKAGE/'upstream_attention.py').read_bytes(),before)
        self.assertNotIn(self.module.CEIL_OLD.encode(),candidate)
        with self.assertRaises(ValueError):self.module.candidate_attention_source(candidate)


if __name__=='__main__':unittest.main()
