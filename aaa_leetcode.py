# TWO SUMS:
# https://leetcode.com/problems/two-sum/

class Solution(object):
    def twoSum(self, nums, target):
        d={}
        for i,num in list(enumerate(nums)):
            print(list(enumerate(nums)))
            print(target-num)
            if target-num in d:
                print(i)
                print(d)
                print(d[target-num])
                return d[target-num], i
            d[num]=i

# # INPUT:
# [2,7,11,15]
# 9

# # OUTPUT:
# [(0, 2), (1, 7), (2, 11), (3, 15)]
# 7
# [(0, 2), (1, 7), (2, 11), (3, 15)]
# 2
# 1
# {2: 0}
# 0

# # FINAL OUTPUT:
# (0,1)

# https://leetcode.com/problems/palindrome-number/

# Input: x = 121
# Output: true

class Solution(object):
    def isPalindrome(self, x):
        if(x>=0):
            y = str(x)
            for i in range(0, len(y)/2):
                if y[i] != y[len(y)-1-i]:
                    return False
            return True
        
        
# My Initial solution:
class Solution(object):
    def isPalindrome(self, x):
        if x>=0:
            x = str(x)
            if len(x)%2 == 0:
                first = x[:len(x)/2]
                second = x[len(x)/2:]
            else:
                first = x[:len(x)/2 +1]
                second = x[len(x)/2:]
            print(first)
            print(second)
            first_list = []
            second_list = []
            for i in first:
                first_list.append(i)
            for i in second:
                second_list.append(i)
            second_list.reverse()
            print(first_list)
            print(second_list)
            if first_list == second_list:
                return True
            else:
                return False
        else:
            return False
        
        
