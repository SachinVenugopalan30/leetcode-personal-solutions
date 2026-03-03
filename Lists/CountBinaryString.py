# 696. Count Binary Substrings
# https://leetcode.com/problems/count-binary-substrings

# intuition - instead of looking at each individual string
# start looking at groups of strings and count the total of them till we reach another character
# add the count to the group, set counter to 1 for the new character
# then, iterate through the groups and take the min since thats the max
# number of substrings we can make with the groups
# return the sum of the final list
class Solution:
    def countBinarySubstrings(self, s: str) -> int:
        groups = []
        counter = 1  # Start at 1 because the first char exists
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                counter += 1
            else:
                groups.append(counter)
                counter = 1  # reset counter to 1 for the newest group
        groups.append(counter)  # append the last counter to the group
        res_list = []
        for i in range(len(groups) - 1):
            res_list.append(min(groups[i], groups[i + 1]))
        return sum(res_list)
