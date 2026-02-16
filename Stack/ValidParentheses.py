# 20. Valid Parentheses
# https://leetcode.com/problems/valid-parentheses/


class Solution:
    def isValid(self, s: str) -> bool:
        opening_brackets = ["[", "(", "{"]
        sstack = []
        if len(s) < 2:
            return False
        for bracket in s:
            if bracket in opening_brackets:
                sstack.append(bracket)
            else:
                if len(sstack) == 0:
                    return False
                popped_element = sstack.pop()
                if popped_element == "[" and bracket != "]":
                    return False
                elif popped_element == "(" and bracket != ")":
                    return False
                elif popped_element == "{" and bracket != "}":
                    return False
        return len(sstack) == 0
