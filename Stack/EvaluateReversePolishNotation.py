# 150. Evaluate Reverse Polish Notation
# https://leetcode.com/problems/evaluate-reverse-polish-notation/


class Solution:
    def evaluate_expression(self, term1, term2, op) -> int:
        if op == "+":
            return term1 + term2
        elif op == "-":
            # because the order is reversed here we need to subtract term1 from term2
            return term2 - term1
        elif op == "*":
            return term1 * term2
        else:
            try:
                # same reason here
                res = term2 / term1
                return int(res)
            except ZeroDivisionError:
                return 0

    def evalRPN(self, tokens: List[str]) -> int:
        arith_ops = ["+", "-", "*", "/"]
        ops_stack = []
        for token in tokens:
            if token not in arith_ops:
                ops_stack.append(int(token))
            else:
                term1 = ops_stack.pop()
                term2 = ops_stack.pop()
                evaluated_expression = self.evaluate_expression(term1, term2, token)
                ops_stack.append(evaluated_expression)
        return int(ops_stack.pop())
