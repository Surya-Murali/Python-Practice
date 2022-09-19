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
        
        
# VALID PARANTHESIS - EASY!
#https://leetcode.com/problems/valid-parentheses/submissions/

# Assign a dict - key, value for brackets
# Understand STACK - LIFO (Last In First Out) - Its a type of list
# Dont forget STACK.POP()
# dict[stack.pop()] - to get the value of the corresponding key

class Solution(object):
    def isValid(self, s):
        if len(s)%2 != 0:
            return False
        dict = {'(':')', '{': '}', '[': ']'}
        stack = []
        for i in s:
            if i in dict.keys():
                stack.append(i)
            else:
                if len(stack) > 0:
                    if i != dict[stack.pop()]:
                        return False
                else:
                    return False
        if len(stack)!=0:
            return False
        else:
            return True

# https://leetcode.com/problems/palindrome-number/submissions/
# Reverse a string or Integer:
# Hint: 
# Use pop() - removes last element from the list
# For loop should have range then
# ''.join(l2) -> convert list to String!

class Solution(object):
    def isPalindrome(self, x):
        l1 = list(str(x))
        l2 = []
        for i in range(0, len(l1)):
            l2.append(l1.pop())
        if ''.join(l2) == str(x):
            return True
        else:
            return False
        
# Reversing without String conversion:
# Get the last digit
# Then multiply by 10 , add the last digit and so on
# Divide the number by 10 at the end before getting the last digit
# While loop needed - while number not equal to 0

class Solution(object):
    def isPalindrome(self, x):
        reversed_num = 0
        while x != 0:
            print("X Value: ", x)
            digit = x % 10
            reversed_num = reversed_num * 10 + digit
            print("Rev: ", reversed_num)
            x = x/10
        print(reversed_num)

# Finding out that its not a palindrome early!
class Solution(object):
    def isPalindrome(self, x):
         if(x>=0):
            y = str(x)
            for i in range(0, len(y)/2):
                if y[i] != y[len(y)-1-i]:
                    return False
            return True
        
# Remove duplicates from a list
# https://leetcode.com/problems/remove-duplicates-from-sorted-array/
# Used this code because its non-decreasing

# List Functions
# Remove Duplicate

#(1) Using 'set' - But order is not maintained

l = [1,9,2,1,2]
list(set(l))
# [1, 2, 9]
list(set('azyxbc'))
# ['a', 'c', 'b', 'y', 'x', 'z']

#(2) Using 'dict.fromkeys' - Again order is not maintained
list(dict.fromkeys(l))
# [1, 2, 9]

#(3) Using 'OrderedDict' - Order is maintained
from collections import OrderedDict
list(OrderedDict.fromkeys(l))
# [1, 9, 2]

# OrderedDict and dict.fromkeys also work on strings
OrderedDict.fromkeys('azyxbc')
# ['a', 'z', 'y', 'x', 'b', 'c']

# Sorting of Lists
l = [1, 9, 2, 1, 2]

# Ascending Sort:
l.sort()

# Descending Sort:
l.sort(reverse=True)
l.reverse()

