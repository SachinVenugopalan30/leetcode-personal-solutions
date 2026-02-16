class MinStack:
    def __init__(self):
        self.main_stack = []
        self.min_stack = []

    def push(self, val: int) -> None:
        self.main_stack.append(val)
        if len(self.min_stack) == 0:
            self.min_stack.append(val)
        else:
            if self.min_stack[-1] > val:
                self.min_stack.append(val)
            else:
                self.min_stack.append(self.min_stack[-1])
        return None

    def pop(self) -> None:
        if len(self.main_stack) and len(self.min_stack):
            self.main_stack.pop()
            self.min_stack.pop()
        return None

    def top(self) -> int:
        if len(self.main_stack):
            return self.main_stack[-1]

    def getMin(self) -> int:
        if len(self.min_stack):
            return self.min_stack[-1]
