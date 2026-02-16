# 271. Encode and Decode Strings
# Note: Leetcode Premium Question, but free to view and practice in Neetcode 150 - https://neetcode.io/problems/string-encode-and-decode/question


class Solution:
    def encode(self, strs: List[str]) -> str:
        encoded_str = ""
        # intuition: instead of just encoding with a special character
        # we encode with the length of each word followed by the special character
        # we do this because if we just use the special character as it is,
        # it might appear in the actual string and cause ambiguity
        for single_str in strs:
            tmp_str = str(len(single_str)) + "#" + single_str
            encoded_str += tmp_str
        return encoded_str

    def decode(self, s: str) -> List[str]:
        decoded_list = []
        i = 0
        while i < len(s):
            j = i
            while s[j] != "#":
                j += 1
            k = int(s[i:j])
            word = s[j + 1 : j + 1 + k]
            decoded_list.append(word)
            i = j + 1 + k
        return decoded_list
