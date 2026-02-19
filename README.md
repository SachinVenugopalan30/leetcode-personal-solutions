## I hate leetcode and so do you

Anyway here are the solutions to the problems I've solved. I've even included explanantions in some of them and split them into concepts. Have fun or whatever.

# DSA Patterns Study Plan

> **⬇️ Jump to: [Progress Checklist](#-progress-checklist)**

---

## How to Identify What Kind of Problem You're Looking At

Before diving into any problem, you need a mental framework for *classifying* it. This is the single most important meta-skill in DSA. When you read a problem, don't immediately think about code — think about *shape*. Ask yourself these questions in order: What is the input? (Array, string, linked list, tree, graph, matrix?) What is the output? (A single value, a boolean, a subset, an index, a transformed structure?) What is the constraint? (Contiguous subarray, sorted order, cycle detection, optimization under limits, all possible combinations?) The constraint is the biggest clue. If the problem says "contiguous subarray" or "substring," think Sliding Window or Prefix Sum. If it says "sorted array," think Two Pointers or Binary Search. If it says "maximum/minimum under a weight limit," think Knapsack. If it says "all permutations/combinations," think Backtracking. If it says "next greater/smaller element," think Monotonic Stack. If it says "shortest path" or "level-by-level," think BFS. If it says "explore all paths" or "connected components," think DFS.

Also pay attention to the *size constraints*. If `n` is up to 10^4 or 10^5, an O(n log n) or O(n) solution is expected. If `n` is small (say ≤ 20), you can often brute-force with backtracking or bitmask DP. If `n` is up to 10^6 or larger, you almost certainly need O(n) or O(n log n).

The thought process is: **Read → Classify the shape → Recall the pattern → Apply the template → Handle edge cases.** Over time, this becomes instinct.

---

## Pattern 1: Two Pointers

### What It Is

The two pointers technique uses two indices that traverse a data structure — usually an array or string — from different positions or at different speeds. The most common setups are: one pointer at the start and one at the end moving inward, or both starting from the beginning with one moving faster.

### When to Use It

Use two pointers when the input is sorted (or can be sorted without losing information), when you're looking for pairs or triplets that satisfy some condition, when you need to partition an array in-place, or when you're comparing characters from both ends (like palindrome checks).

### The Intuition

The key insight is that by maintaining two pointers, you can eliminate large portions of the search space in a single step. In a brute-force approach you'd check every pair (O(n²)), but with two pointers on a sorted array, each step either advances the left pointer or retreats the right pointer, giving you O(n).

### Template

```python
def two_pointer_opposite(arr, target):
    """Two pointers moving inward from both ends."""
    left, right = 0, len(arr) - 1
    while left < right:
        current = arr[left] + arr[right]
        if current == target:
            return [left, right]
        elif current < target:
            left += 1       # need a bigger sum
        else:
            right -= 1      # need a smaller sum
    return []

def two_pointer_same_direction(arr):
    """Two pointers moving in the same direction (e.g., remove duplicates)."""
    slow = 0
    for fast in range(1, len(arr)):
        if arr[fast] != arr[slow]:
            slow += 1
            arr[slow] = arr[fast]
    return slow + 1  # length of unique portion
```

### Key Problems to Practice

- Two Sum II – Input Array Is Sorted
- 3Sum
- Container With Most Water
- Remove Duplicates from Sorted Array
- Trapping Rain Water

---

## Pattern 2: Fast & Slow Pointers

### What It Is

Also called the Tortoise and Hare algorithm. Two pointers traverse a structure at different speeds — typically one moves one step at a time and the other moves two steps. They are primarily used with linked lists.

### When to Use It

Use this when you need to detect a cycle in a linked list, find the middle of a linked list, find the start of a cycle, or detect duplicates in an array that can be modeled as a linked list (like Floyd's algorithm applied to Find the Duplicate Number).

### The Intuition

If there's a cycle, the fast pointer will eventually "lap" the slow pointer, and they'll meet inside the cycle. If there's no cycle, the fast pointer will reach the end. To find the *start* of the cycle, after detection, reset one pointer to the head and move both at the same speed — they meet at the cycle's entrance. For finding the middle: when fast reaches the end, slow is at the midpoint because it moved half the distance.

### Template

```python
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

def find_cycle_start(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            # Reset one pointer to head
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow  # start of cycle
    return None

def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
```

### Key Problems to Practice

- Linked List Cycle II
- Remove Nth Node From End of List
- Find the Duplicate Number
- Palindrome Linked List
- Happy Number

---

## Pattern 3: Sliding Window

### What It Is

A sliding window maintains a "window" — a contiguous subarray or substring — and slides it across the data. Instead of recomputing everything from scratch for each position, you update the window incrementally by adding the new element and removing the old one.

### When to Use It

Use this when the problem asks about contiguous subarrays or substrings, when you need to find a maximum/minimum length subarray satisfying some property, or when you see phrases like "at most K distinct characters," "subarray with sum ≥ target," or "longest/shortest substring containing."

### The Intuition

The brute force for subarray problems is O(n²) — check every possible start/end pair. The sliding window reduces this to O(n) by recognizing that as the window slides right, you only need to adjust the boundaries. If adding the rightmost element breaks the condition, shrink from the left. If the condition is satisfied, try to expand from the right. You're essentially maintaining an invariant about what's inside the window.

There are two flavors:
- **Fixed-size window**: The window size `k` is given. Slide it one step at a time.
- **Variable-size window**: Expand the right boundary until the condition breaks, then shrink the left boundary until the condition is restored.

### Template

```python
def sliding_window_variable(s):
    """Variable-size window template."""
    left = 0
    window = {}  # or a counter, set, running sum, etc.
    result = 0

    for right in range(len(s)):
        # Expand: add s[right] to window state
        window[s[right]] = window.get(s[right], 0) + 1

        # Shrink: while the window violates the condition
        while window_is_invalid(window):
            window[s[left]] -= 1
            if window[s[left]] == 0:
                del window[s[left]]
            left += 1

        # Update result (longest valid window)
        result = max(result, right - left + 1)

    return result

def sliding_window_fixed(arr, k):
    """Fixed-size window template."""
    window_sum = sum(arr[:k])
    max_sum = window_sum

    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)

    return max_sum
```

### Key Problems to Practice

- Longest Substring Without Repeating Characters
- Minimum Window Substring
- Longest Repeating Character Replacement
- Permutation in String
- Maximum Average Subarray I
- Minimum Size Subarray Sum

---

## Pattern 4: Prefix Sum / Prefix Product

### What It Is

Prefix sums (or products) precompute cumulative values up to each index. After building the prefix array, you can answer any range query (sum of elements from index `i` to `j`) in O(1) time.

### When to Use It

Use this when you need to answer multiple range sum queries, find subarrays with a specific sum, count subarrays satisfying sum conditions, or compute running totals efficiently. The classic signal is: "subarray sum equals K."

### The Intuition

The sum of elements from index `i` to `j` is `prefix[j+1] - prefix[i]`. By storing prefix sums, you convert a range sum problem into a simple subtraction. When combined with a hash map (storing prefix sums you've seen and their counts), you can find subarrays with a target sum in O(n) — because if `prefix[j] - prefix[i] = target`, then there exists a subarray summing to `target`.

### Template

```python
def build_prefix_sum(arr):
    prefix = [0] * (len(arr) + 1)
    for i in range(len(arr)):
        prefix[i + 1] = prefix[i] + arr[i]
    return prefix

def range_sum(prefix, i, j):
    """Sum of arr[i..j] inclusive."""
    return prefix[j + 1] - prefix[i]

def subarray_sum_equals_k(arr, k):
    """Count subarrays whose sum equals k."""
    count = 0
    current_sum = 0
    prefix_counts = {0: 1}  # empty prefix has sum 0

    for num in arr:
        current_sum += num
        # If (current_sum - k) was a previous prefix sum,
        # then the subarray between that point and here sums to k
        if current_sum - k in prefix_counts:
            count += prefix_counts[current_sum - k]
        prefix_counts[current_sum] = prefix_counts.get(current_sum, 0) + 1

    return count
```

### Key Problems to Practice

- Subarray Sum Equals K
- Range Sum Query – Immutable
- Product of Array Except Self
- Contiguous Array
- Find Pivot Index

---

## Pattern 5: Merge Intervals

### What It Is

Interval problems involve ranges defined by a start and end value. The core operations are merging overlapping intervals, inserting a new interval, or finding gaps/conflicts.

### When to Use It

Use this when the problem involves time ranges, scheduling, meeting rooms, or any data with [start, end] pairs. Keywords: "overlapping," "merge," "conflict," "non-overlapping," "minimum number of arrows/groups."

### The Intuition

Almost always, the first step is to **sort intervals by their start time**. Once sorted, you process them left to right. Two intervals overlap if the current interval's start is less than or equal to the previous interval's end. When they overlap, you merge them by extending the end to the maximum of both ends. If they don't overlap, the previous interval is finalized and you move on.

The mental model: imagine intervals as colored bars on a timeline. Sorting aligns them left to right, and then you just check if each new bar touches or overlaps the previous one.

### Template

```python
def merge_intervals(intervals):
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        if start <= merged[-1][1]:   # overlapping
            merged[-1][1] = max(merged[-1][1], end)
        else:                         # non-overlapping
            merged.append([start, end])

    return merged

def can_attend_all(intervals):
    """Check if any intervals overlap (e.g., meeting room conflict)."""
    intervals.sort(key=lambda x: x[0])
    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i - 1][1]:
            return False
    return True
```

### Key Problems to Practice

- Merge Intervals
- Insert Interval
- Non-overlapping Intervals
- My Calendar II
- Minimum Number of Arrows to Burst Balloons
- Meeting Rooms / Meeting Rooms II

---

## Pattern 6: Top K Elements

### What It Is

This pattern finds the K largest, K smallest, or K most frequent elements from a dataset. It uses a heap (priority queue) or the quickselect algorithm.

### When to Use It

Any time you see "K largest," "K smallest," "K most frequent," "K closest," or "top K." Also relevant when you need to maintain a running top-K as data streams in.

### The Intuition

A brute force approach sorts the entire array (O(n log n)) and takes the top K. A heap-based approach is smarter: maintain a min-heap of size K. As you process each element, push it onto the heap. If the heap exceeds size K, pop the smallest. At the end, the heap contains exactly the K largest elements. This runs in O(n log k), which is better when k << n.

For K most frequent: first count frequencies with a hash map, then use a heap to extract the top K frequencies. Alternatively, use **bucket sort** — create an array where index `i` holds all elements with frequency `i`, then iterate from the back.

### Template

```python
import heapq

def top_k_largest(nums, k):
    """Find the K largest elements using a min-heap of size k."""
    min_heap = []
    for num in nums:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)  # remove smallest
    return min_heap  # contains K largest

def top_k_frequent(nums, k):
    """Find the K most frequent elements."""
    from collections import Counter
    counts = Counter(nums)
    # Use a min-heap of size k on (frequency, element)
    return heapq.nlargest(k, counts.keys(), key=counts.get)

def kth_largest(nums, k):
    """Find the Kth largest element."""
    min_heap = []
    for num in nums:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)
    return min_heap[0]  # smallest in heap = Kth largest overall
```

### Key Problems to Practice

- Kth Largest Element in an Array
- Top K Frequent Elements
- K Closest Points to Origin
- Sort Characters by Frequency
- Find K Pairs with Smallest Sums

---

## Pattern 7: K-Way Merge

### What It Is

This pattern merges K sorted lists (or arrays, streams, etc.) into a single sorted output. It uses a min-heap to always pick the smallest available element across all K sources.

### When to Use It

Whenever you see "merge K sorted" anything — sorted lists, sorted arrays, sorted streams. Also useful for problems like "find the smallest range covering elements from K lists."

### The Intuition

You can't just concatenate and sort (that's O(N log N) where N is the total number of elements). Instead, maintain a min-heap with one element from each list. Pop the smallest, then push the next element from that same list. This always gives you the globally smallest available element. The heap never exceeds size K, so each operation is O(log k).

### Template

```python
import heapq

def merge_k_sorted_lists(lists):
    """Merge k sorted lists into one sorted list."""
    min_heap = []
    # Initialize: push the first element of each list
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(min_heap, (lst[0], i, 0))
            # (value, list_index, element_index)

    result = []
    while min_heap:
        val, list_idx, elem_idx = heapq.heappop(min_heap)
        result.append(val)
        # Push the next element from the same list
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(min_heap, (next_val, list_idx, elem_idx + 1))

    return result
```

### Key Problems to Practice

- Merge K Sorted Lists
- Kth Smallest Element in a Sorted Matrix
- Smallest Range Covering Elements from K Lists
- Find K Pairs with Smallest Sums

---

## Pattern 8: Two Heaps

### What It Is

This pattern maintains two heaps simultaneously — typically a max-heap for the lower half of the data and a min-heap for the upper half. This lets you efficiently track the median of a dynamically changing dataset.

### When to Use It

Finding the running median, problems where you need to efficiently access both the maximum of the smaller half and the minimum of the larger half of a dataset, or any problem where you're partitioning a stream into two balanced halves.

### The Intuition

The median splits data into two halves. A max-heap gives you instant access to the largest element of the lower half, and a min-heap gives instant access to the smallest element of the upper half. By keeping the heaps balanced (their sizes differ by at most 1), the median is either the top of the larger heap (odd total) or the average of both tops (even total).

### Template

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.lo = []  # max-heap (store negatives)
        self.hi = []  # min-heap

    def add_num(self, num):
        # Always add to max-heap first
        heapq.heappush(self.lo, -num)
        # Balance: move largest of lo to hi
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        # Ensure lo has >= elements as hi
        if len(self.lo) < len(self.hi):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def find_median(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2.0
```

### Key Problems to Practice

- Find Median from Data Stream
- Sliding Window Median
- IPO (Maximize Capital)

---

## Pattern 9: Monotonic Stack

### What It Is

A monotonic stack is a stack that maintains its elements in strictly increasing or decreasing order. As you push new elements, you pop any elements that violate the monotonic property. This is particularly useful for "next greater element" and "next smaller element" style problems.

### When to Use It

Use this when you need to find the next greater/smaller element for each element, when computing areas of histograms, when dealing with stock span problems, or any problem that asks about the nearest element satisfying a comparison condition.

### The Intuition

Think of it this way: you're processing elements left to right. For each new element, any previous elements in the stack that are *smaller* (for "next greater" problems) have now found their answer — the current element is their next greater element. So you pop them and record the answer, then push the current element. The stack always holds elements that are still "waiting" for their answer.

### Template

```python
def next_greater_element(nums):
    """For each element, find the next element that is greater."""
    n = len(nums)
    result = [-1] * n
    stack = []  # stores indices

    for i in range(n):
        # Pop elements that have found their next greater
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)

    return result

def daily_temperatures(temps):
    """How many days until a warmer temperature?"""
    n = len(temps)
    result = [0] * n
    stack = []

    for i in range(n):
        while stack and temps[stack[-1]] < temps[i]:
            prev = stack.pop()
            result[prev] = i - prev
        stack.append(i)

    return result
```

### Key Problems to Practice

- Daily Temperatures
- Next Greater Element I / II
- Largest Rectangle in Histogram
- Trapping Rain Water (monotonic stack approach)
- Online Stock Span

---

## Pattern 10: Modified Binary Search

### What It Is

Standard binary search finds a target in a sorted array in O(log n). Modified binary search extends this idea to rotated arrays, unsorted-but-structured data, or — most powerfully — **binary search on the answer space**, where you binary-search over possible answer values rather than array indices.

### When to Use It

Use binary search when you see a sorted array (possibly rotated), when you need to find a boundary (first/last occurrence), or when the problem asks to "minimize the maximum" or "find the minimum that satisfies a condition" — these are classic binary search on answer problems.

### The Intuition

Binary search works whenever there's a **monotonic predicate**: a condition that is `False` for all values below some threshold and `True` for all values above it (or vice versa). You're searching for the boundary where the predicate flips. This applies far beyond sorted arrays — any problem where you can frame "is answer X feasible?" as a yes/no function that flips at some point can use binary search.

### Template

```python
def binary_search_standard(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

def binary_search_on_answer(lo, hi, is_feasible):
    """Find the minimum value in [lo, hi] for which is_feasible returns True."""
    while lo < hi:
        mid = (lo + hi) // 2
        if is_feasible(mid):
            hi = mid       # mid might be the answer, search left
        else:
            lo = mid + 1   # mid is too small
    return lo

# Example: Minimum days to make m bouquets
# is_feasible checks: "can we make m bouquets if we wait `days` days?"
# Binary search over possible values of `days`.
```

### Key Problems to Practice

- Search in Rotated Sorted Array
- Find Minimum in Rotated Sorted Array
- Koko Eating Bananas
- Capacity to Ship Packages Within D Days
- Split Array Largest Sum
- Find Peak Element

---

## Pattern 11: BFS (Breadth-First Search)

### What It Is

BFS explores a graph or tree level by level, using a queue. It visits all neighbors of the current node before moving to their neighbors.

### When to Use It

Use BFS when you need the **shortest path** in an unweighted graph, when you need to process things level by level (like level-order traversal of a tree), or when the problem involves minimum number of steps/moves/transformations.

### The Intuition

BFS guarantees that the first time you reach a node, you've reached it via the shortest path (in unweighted graphs). Think of it as a "ripple" expanding outward from the source. Each "ring" of the ripple represents nodes at the same distance. This is why BFS is the go-to for shortest path problems in unweighted settings.

### Template

```python
from collections import deque

def bfs_shortest_path(graph, start, end):
    queue = deque([(start, 0)])  # (node, distance)
    visited = {start}

    while queue:
        node, dist = queue.popleft()
        if node == end:
            return dist
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))

    return -1  # unreachable

def bfs_level_order(root):
    """Level-order traversal of a binary tree."""
    if not root:
        return []
    result = []
    queue = deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result
```

### Key Problems to Practice

- Binary Tree Level Order Traversal
- Rotting Oranges
- Word Ladder
- Shortest Path in Binary Matrix
- 01 Matrix
- Open the Lock

---

## Pattern 12: DFS (Depth-First Search)

### What It Is

DFS explores as deep as possible along each branch before backtracking. It can be implemented recursively (using the call stack) or iteratively (using an explicit stack).

### When to Use It

Use DFS for exploring all paths in a graph/tree, detecting cycles, finding connected components, topological sorting, or any problem that requires exhaustive exploration of possibilities.

### The Intuition

DFS follows one path until it hits a dead end, then backtracks to the last decision point and tries the next option. In trees, the three classic orderings — preorder, inorder, postorder — are all DFS variants. In graphs, DFS naturally handles cycle detection (if you revisit a node that's in the current recursion stack, there's a cycle).

### Template

```python
def dfs_graph(graph, start):
    """Explore all reachable nodes."""
    visited = set()

    def dfs(node):
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

    dfs(start)
    return visited

def dfs_tree_traversals(root):
    """All three DFS orderings."""
    def preorder(node):
        if not node: return
        print(node.val)        # process BEFORE children
        preorder(node.left)
        preorder(node.right)

    def inorder(node):
        if not node: return
        inorder(node.left)
        print(node.val)        # process BETWEEN children
        inorder(node.right)

    def postorder(node):
        if not node: return
        postorder(node.left)
        postorder(node.right)
        print(node.val)        # process AFTER children

def count_islands(grid):
    """Classic DFS on a matrix."""
    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0':
            return
        grid[r][c] = '0'  # mark visited
        for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
            dfs(r + dr, c + dc)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                dfs(r, c)
                count += 1
    return count
```

### Key Problems to Practice

- Number of Islands
- Clone Graph
- Course Schedule (cycle detection)
- Path Sum / Path Sum II
- Validate Binary Search Tree
- Surrounded Regions

---

## Pattern 13: Backtracking

### What It Is

Backtracking is a refinement of DFS where you build a solution incrementally, and abandon ("prune") a path as soon as you determine it can't lead to a valid solution. It's DFS with early termination.

### When to Use It

Use backtracking when the problem asks for all possible solutions (permutations, combinations, subsets), when you need to satisfy constraints (N-Queens, Sudoku), or when the problem involves making a series of choices where each choice constrains future choices.

### The Intuition

Think of it as exploring a decision tree. At each node, you have several choices. You make a choice, recurse deeper, then **undo the choice** (backtrack) and try the next option. The "undo" step is what distinguishes backtracking from plain DFS. The power comes from pruning: if you can detect early that the current path is a dead end, you skip entire subtrees.

The universal template: choose → explore → unchoose.

### Template

```python
def backtrack(candidates, target):
    result = []

    def helper(start, current, remaining):
        if remaining == 0:
            result.append(current[:])  # found a valid solution
            return
        if remaining < 0:
            return                      # pruning: overshoot

        for i in range(start, len(candidates)):
            # Skip duplicates (if needed)
            if i > start and candidates[i] == candidates[i - 1]:
                continue

            current.append(candidates[i])       # choose
            helper(i + 1, current, remaining - candidates[i])  # explore
            current.pop()                        # unchoose (backtrack)

    candidates.sort()  # sort for duplicate handling
    helper(0, [], target)
    return result

def permutations(nums):
    result = []

    def helper(current):
        if len(current) == len(nums):
            result.append(current[:])
            return
        for num in nums:
            if num not in current:  # simple check (or use a visited set)
                current.append(num)
                helper(current)
                current.pop()

    helper([])
    return result
```

### Key Problems to Practice

- Subsets / Subsets II
- Permutations / Permutations II
- Combination Sum / Combination Sum II
- N-Queens
- Palindrome Partitioning
- Letter Combinations of a Phone Number
- Word Search

---

## Pattern 14: 0/1 Knapsack (Dynamic Programming)

### What It Is

The classic optimization DP pattern. You have a set of items, each with a weight and value. You have a capacity limit. Each item can be taken once (0/1 — take it or leave it). The goal is to maximize total value without exceeding the capacity.

### When to Use It

Whenever you see a problem that involves choosing from a set of items with constraints (budget, weight, capacity) and you want to maximize or minimize something. Also applies to subset sum, partition equal subset sum, and target sum problems.

### The Intuition

For each item, you have two choices: include it or exclude it. If you include it, your remaining capacity decreases. The key DP idea is: `dp[i][w]` = the maximum value achievable using the first `i` items with capacity `w`. The recurrence is:
- If you skip item `i`: `dp[i][w] = dp[i-1][w]`
- If you take item `i`: `dp[i][w] = dp[i-1][w - weight[i]] + value[i]`
- Take the max of both.

This can be space-optimized to a 1D array by iterating capacity in reverse.

### Template

```python
def knapsack_01(weights, values, capacity):
    """Classic 0/1 knapsack, space-optimized."""
    n = len(weights)
    dp = [0] * (capacity + 1)

    for i in range(n):
        # Iterate in REVERSE to avoid using item i twice
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])

    return dp[capacity]

def can_partition(nums):
    """Partition Equal Subset Sum — a 0/1 knapsack variant."""
    total = sum(nums)
    if total % 2 != 0:
        return False
    target = total // 2
    dp = [False] * (target + 1)
    dp[0] = True

    for num in nums:
        for t in range(target, num - 1, -1):
            dp[t] = dp[t] or dp[t - num]

    return dp[target]
```

### Key Problems to Practice

- Partition Equal Subset Sum
- Target Sum
- Last Stone Weight II
- Ones and Zeroes

---

## Pattern 15: Unbounded Knapsack

### What It Is

A variant of the knapsack where each item can be used **unlimited times**. The structure is similar, but the iteration direction changes.

### When to Use It

Coin change problems, cutting rod problems, or any optimization where you can reuse the same item multiple times.

### The Intuition

The only difference from 0/1 knapsack: when considering item `i`, you iterate capacity **forward** (not reverse), because you *want* to allow reusing the same item. Forward iteration means `dp[w - weight[i]]` might already include item `i`, which is exactly the behavior you want.

### Template

```python
def coin_change(coins, amount):
    """Minimum coins to make the amount (unbounded knapsack)."""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        # Forward iteration: allows reuse of same coin
        for a in range(coin, amount + 1):
            dp[a] = min(dp[a], dp[a - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1

def coin_change_ways(coins, amount):
    """Count number of ways to make the amount."""
    dp = [0] * (amount + 1)
    dp[0] = 1

    for coin in coins:
        for a in range(coin, amount + 1):
            dp[a] += dp[a - coin]

    return dp[amount]
```

### Key Problems to Practice

- Coin Change
- Coin Change II
- Perfect Squares
- Minimum Cost for Tickets

---

## Pattern 16: Longest Increasing Subsequence (LIS)

### What It Is

Finding the longest subsequence (not necessarily contiguous) of a given sequence where elements are in strictly ascending order. This is a foundational DP pattern.

### When to Use It

Any problem about longest increasing/decreasing subsequences, building envelopes, scheduling jobs by increasing order, or chain-like optimization problems.

### The Intuition

The basic DP approach: `dp[i]` = length of the LIS ending at index `i`. For each `i`, check all `j < i` — if `nums[j] < nums[i]`, then `dp[i] = max(dp[i], dp[j] + 1)`. This is O(n²).

The optimized approach uses **patience sorting**: maintain a list `tails` where `tails[i]` is the smallest tail element among all increasing subsequences of length `i+1`. For each new element, binary search for its position in `tails`. This gives O(n log n).

### Template

```python
def lis_dp(nums):
    """O(n²) DP approach."""
    n = len(nums)
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)

def lis_binary_search(nums):
    """O(n log n) using patience sorting."""
    import bisect
    tails = []

    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)    # extends longest subsequence
        else:
            tails[pos] = num     # replace to keep tails small

    return len(tails)
```

### Key Problems to Practice

- Longest Increasing Subsequence
- Russian Doll Envelopes
- Maximum Length of Pair Chain
- Number of Longest Increasing Subsequence
- Longest String Chain

---

## Pattern 17: DP on Matrices / Grids

### What It Is

Dynamic programming where the state space is a 2D grid. The solution at each cell depends on solutions at neighboring cells (usually the cell above and the cell to the left).

### When to Use It

Pathfinding in grids (unique paths, minimum cost paths), string comparison problems (edit distance, longest common subsequence), or any problem that naturally maps to a 2D table.

### The Intuition

Think of filling in a table cell by cell. Each cell `dp[i][j]` is computed from previously computed cells. The key is defining what `dp[i][j]` *means* and what the recurrence is. For grid paths: `dp[i][j]` = number of ways to reach cell (i,j). For LCS: `dp[i][j]` = LCS of the first `i` characters of string A and first `j` characters of string B.

### Template

```python
def unique_paths(m, n):
    """Count unique paths from top-left to bottom-right."""
    dp = [[1] * n for _ in range(m)]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]

    return dp[m-1][n-1]

def min_path_sum(grid):
    """Minimum sum path from top-left to bottom-right."""
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]

    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])

    return dp[m-1][n-1]

def longest_common_subsequence(text1, text2):
    """Classic 2D DP on two strings."""
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]
```

### Key Problems to Practice

- Unique Paths / Unique Paths II
- Minimum Path Sum
- Longest Common Subsequence
- Edit Distance
- Maximal Square
- Dungeon Game

---

## Recommended Study Order

The patterns above are ordered roughly by difficulty and dependency. Here's a suggested study plan:

**Week 1–2: Array Fundamentals**
Two Pointers → Sliding Window → Prefix Sum. These three patterns form the bedrock. They show up everywhere and teach you to think about optimizing brute-force array solutions.

**Week 3: Linked List Patterns**
Fast & Slow Pointers. Shorter section, but essential for interviews. Practice pointer manipulation until it's second nature.

**Week 4: Sorting-Adjacent Patterns**
Merge Intervals → Modified Binary Search. Both rely on sorted data and teach you to think about ordered structures.

**Week 5: Heap Patterns**
Top K Elements → K-Way Merge → Two Heaps. Master the heap API in your language. These problems are extremely common in real interviews.

**Week 6: Stack Patterns**
Monotonic Stack. A focused topic but appears frequently. Understand *why* the stack maintains its ordering.

**Week 7–8: Graph Traversal**
BFS → DFS → Backtracking. Start with tree problems (simpler graphs), then move to general graphs and grid problems. Backtracking is DFS with extra pruning.

**Week 9–10: Dynamic Programming**
0/1 Knapsack → Unbounded Knapsack → LIS → DP on Matrices. DP is the hardest section. Start by understanding the recurrence in plain English before writing code. Draw the DP table by hand for small examples.

**Ongoing: Mixed Practice**
After completing all patterns, do timed mixed problem sets. The real skill is *identifying* which pattern applies, not just executing the pattern once you know which one it is.

---

## Final Thoughts on Problem-Solving Mindset

The gap between "knowing patterns" and "solving problems" is bridged by practice. When you see a new problem, resist the urge to code immediately. Spend the first few minutes thinking: What pattern does this look like? What's the brute force? Why is it slow? What structure can I exploit to make it faster? Walk through a small example by hand. Only then start coding.

If you're stuck for more than 15–20 minutes, look at the hint or topic tags. Understanding *why* a pattern applies is more valuable than solving the problem independently. Over time, you'll see the same shapes over and over, wrapped in different stories. That's when problems stop being hard and start being familiar.

---

## ✅ Progress Checklist

Use this to track your progress across every pattern and problem. Tick off each item as you complete it.

> **⬆️ Back to: [Top of Document](#dsa-patterns-study-plan)**

---

### Pattern Study Progress

- [ ] Read the "How to Identify Problem Types" section
- [ ] **Pattern 1: Two Pointers** — Read & understood
- [ ] **Pattern 2: Fast & Slow Pointers** — Read & understood
- [ ] **Pattern 3: Sliding Window** — Read & understood
- [ ] **Pattern 4: Prefix Sum / Prefix Product** — Read & understood
- [ ] **Pattern 5: Merge Intervals** — Read & understood
- [ ] **Pattern 6: Top K Elements** — Read & understood
- [ ] **Pattern 7: K-Way Merge** — Read & understood
- [ ] **Pattern 8: Two Heaps** — Read & understood
- [ ] **Pattern 9: Monotonic Stack** — Read & understood
- [ ] **Pattern 10: Modified Binary Search** — Read & understood
- [ ] **Pattern 11: BFS** — Read & understood
- [ ] **Pattern 12: DFS** — Read & understood
- [ ] **Pattern 13: Backtracking** — Read & understood
- [ ] **Pattern 14: 0/1 Knapsack** — Read & understood
- [ ] **Pattern 15: Unbounded Knapsack** — Read & understood
- [ ] **Pattern 16: Longest Increasing Subsequence** — Read & understood
- [ ] **Pattern 17: DP on Matrices / Grids** — Read & understood

---

### Week 1–2: Array Fundamentals

**Two Pointers**
- [ ] Two Sum II – Input Array Is Sorted
- [ ] 3Sum
- [ ] Container With Most Water
- [ ] Remove Duplicates from Sorted Array
- [ ] Trapping Rain Water

**Sliding Window**
- [ ] Longest Substring Without Repeating Characters
- [ ] Minimum Window Substring
- [ ] Longest Repeating Character Replacement
- [ ] Permutation in String
- [ ] Maximum Average Subarray I
- [ ] Minimum Size Subarray Sum

**Prefix Sum**
- [ ] Subarray Sum Equals K
- [ ] Range Sum Query – Immutable
- [ ] Product of Array Except Self
- [ ] Contiguous Array
- [ ] Find Pivot Index

---

### Week 3: Linked List Patterns

**Fast & Slow Pointers**
- [ ] Linked List Cycle II
- [ ] Remove Nth Node From End of List
- [ ] Find the Duplicate Number
- [ ] Palindrome Linked List
- [ ] Happy Number

---

### Week 4: Sorting-Adjacent Patterns

**Merge Intervals**
- [ ] Merge Intervals
- [ ] Insert Interval
- [ ] Non-overlapping Intervals
- [ ] My Calendar II
- [ ] Minimum Number of Arrows to Burst Balloons
- [ ] Meeting Rooms / Meeting Rooms II

**Modified Binary Search**
- [ ] Search in Rotated Sorted Array
- [ ] Find Minimum in Rotated Sorted Array
- [ ] Koko Eating Bananas
- [ ] Capacity to Ship Packages Within D Days
- [ ] Split Array Largest Sum
- [ ] Find Peak Element

---

### Week 5: Heap Patterns

**Top K Elements**
- [ ] Kth Largest Element in an Array
- [ ] Top K Frequent Elements
- [ ] K Closest Points to Origin
- [ ] Sort Characters by Frequency
- [ ] Find K Pairs with Smallest Sums

**K-Way Merge**
- [ ] Merge K Sorted Lists
- [ ] Kth Smallest Element in a Sorted Matrix
- [ ] Smallest Range Covering Elements from K Lists
- [ ] Find K Pairs with Smallest Sums

**Two Heaps**
- [ ] Find Median from Data Stream
- [ ] Sliding Window Median
- [ ] IPO (Maximize Capital)

---

### Week 6: Stack Patterns

**Monotonic Stack**
- [ ] Daily Temperatures
- [ ] Next Greater Element I
- [ ] Next Greater Element II
- [ ] Largest Rectangle in Histogram
- [ ] Trapping Rain Water (stack approach)
- [ ] Online Stock Span

---

### Week 7–8: Graph Traversal

**BFS**
- [ ] Binary Tree Level Order Traversal
- [ ] Rotting Oranges
- [ ] Word Ladder
- [ ] Shortest Path in Binary Matrix
- [ ] 01 Matrix
- [ ] Open the Lock

**DFS**
- [ ] Number of Islands
- [ ] Clone Graph
- [ ] Course Schedule
- [ ] Path Sum / Path Sum II
- [ ] Validate Binary Search Tree
- [ ] Surrounded Regions

**Backtracking**
- [ ] Subsets
- [ ] Subsets II
- [ ] Permutations
- [ ] Permutations II
- [ ] Combination Sum
- [ ] Combination Sum II
- [ ] N-Queens
- [ ] Palindrome Partitioning
- [ ] Letter Combinations of a Phone Number
- [ ] Word Search

---

### Week 9–10: Dynamic Programming

**0/1 Knapsack**
- [ ] Partition Equal Subset Sum
- [ ] Target Sum
- [ ] Last Stone Weight II
- [ ] Ones and Zeroes

**Unbounded Knapsack**
- [ ] Coin Change
- [ ] Coin Change II
- [ ] Perfect Squares
- [ ] Minimum Cost for Tickets

**Longest Increasing Subsequence**
- [ ] Longest Increasing Subsequence
- [ ] Russian Doll Envelopes
- [ ] Maximum Length of Pair Chain
- [ ] Number of Longest Increasing Subsequence
- [ ] Longest String Chain

**DP on Matrices**
- [ ] Unique Paths
- [ ] Unique Paths II
- [ ] Minimum Path Sum
- [ ] Longest Common Subsequence
- [ ] Edit Distance
- [ ] Maximal Square
- [ ] Dungeon Game

---

### Milestones

- [ ] All 17 patterns read and understood
- [ ] At least 3 problems solved per pattern
- [ ] All Easy problems complete
- [ ] All Medium problems complete
- [ ] All Hard problems attempted
- [ ] Completed 5+ timed mixed practice sessions
- [ ] Can identify the correct pattern within 2 minutes for new problems
