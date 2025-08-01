# --------------------------------------
# Type Casting in Python (Explicit Type Conversion)
# --------------------------------------

# Type casting means converting one data type to another.
# Python provides built-in functions: int(), float(), str(), bool(), list(), tuple(), set()

# 1. int() → Converts to integer
print(int("123"))        # ✅ 123 (from string)
print(int(3.99))         # ✅ 3 (truncates decimal)
# print(int("abc"))      # ❌ ValueError: invalid literal
# print(int("3.14"))     # ❌ ValueError: must use float() first

# 2. float() → Converts to float
print(float("3.14"))     # ✅ 3.14
print(float(10))         # ✅ 10.0
# print(float("ten"))    # ❌ ValueError

# 3. str() → Converts to string
print(str(100))          # ✅ "100"
print(str(3.14))         # ✅ "3.14"
print(str(True))         # ✅ "True"
print(str([1, 2, 3]))     # ✅ "[1, 2, 3]"

# 4. bool() → Converts to Boolean (based on truthy/falsy)
print(bool(0))           # False
print(bool(""))          # False
print(bool([]))          # False
print(bool("hello"))     # True
print(bool(123))         # True

# 5. list(), tuple(), set() → Convert iterable types
print(list("abc"))       # ['a', 'b', 'c']
print(tuple([1, 2, 3]))  # (1, 2, 3)
print(set([1, 1, 2]))    # {1, 2}

# --------------------------------------
# Advanced Type Conversion Examples
# --------------------------------------

# Convert float string to int (2 steps)
print(int(float("3.14")))  # 3

# Convert character to ASCII (ord)
print(ord('A'))            # 65

# Convert ASCII to character (chr)
print(chr(65))             # 'A'

# Convert string to unique hash (for IDs, not reversible)
print(hash("xyz"))         # (varies every run)

# --------------------------------------
# Safe Type Casting Using try-except
# --------------------------------------

try:
    val = int("xyz")
except ValueError:
    print("Cannot convert 'xyz' to integer.")

# --------------------------------------
# Summary of Common Conversions

# str → int      → int("123") ✅
# str → float    → float("3.14") ✅
# float → int    → int(4.99) = 4 ✅
# int → str      → str(100) = "100" ✅
# str → list     → list("abc") = ['a','b','c'] ✅
# list → set     → set([1,2,2]) = {1,2} ✅

name = "5"
print(name + "1")
print(int(name) + 1)

n =5
print("".join(str(i) for i in range(n)))