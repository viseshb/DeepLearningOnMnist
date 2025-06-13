# -------------------------------
# Conditionals in Python
# -------------------------------

age = 15

# Basic if-elif-else
if age > 18:
    print("Adult")
elif age < 18:
    print("Teen")
else:
    print("Just an adult")

# Ternary if expression (shorthand)
status = "Adult" if age >= 18 else "Minor"
print("Ternary if result:", status)


# -------------------------------
# Match Statement (Python 3.10+)
# -------------------------------

command = "start"

match command:
    case "start":
        print("System starting...")
    case "stop":
        print("System stopping...")
    case "restart":
        print("System restarting...")
    case _:
        print("Unknown command")


# -------------------------------
# Loops in Python
# -------------------------------

# FOR LOOP
numbers = [10, 20, 30]
for num in numbers:
    print("For loop value:", num)

for i in range(3):  # 0 to 2
    print("Range loop i =", i)

for i in range(1, 6, 2):  # 1, 3, 5
    print("Stepped range:", i)

# WHILE LOOP
count = 0
while count < 3:
    print("While loop count =", count)
    count += 1


# -------------------------------
# Loop Control Statements
# -------------------------------

# break
for i in range(5):
    if i == 3:
        break
    print("Break example:", i)

# continue
for i in range(5):
    if i == 2:
        continue
    print("Continue example:", i)

# Nested loop
for i in range(2):
    for j in range(3):
        print(f"Nested loop: i={i}, j={j}")

# pass
for i in range(3):
    pass  # Placeholder, does nothing


# -------------------------------
# Exception Handling
# -------------------------------

try:
    num = int("xyz")  # This will raise ValueError
    result = 10 / num
except ValueError:
    print("Caught a ValueError: Cannot convert string to int.")
except ZeroDivisionError:
    print("Caught ZeroDivisionError: Division by zero.")
else:
    print("No exception occurred.")
finally:
    print("This block always runs (cleanup, etc).")

# -------------------------------
# Match Statement (Python 3.10+)
# Used for pattern matching, similar to switch-case in other languages
# -------------------------------

# Example 1: Matching exact strings
command = "start"

match command:
    case "start":
        print("System starting...")
    case "stop":
        print("System stopping...")
    case "restart":
        print("System restarting...")
    case _:
        print("Unknown command")

# Example 2: Matching integers with conditions
status_code = 404

match status_code:
    case 200:
        print("OK")
    case 404:
        print("Not Found")
    case 500:
        print("Server Error")
    case _:
        print("Unknown Status Code")

# Example 3: Matching with a tuple
point = (0, 5)

match point:
    case (0, 0):
        print("Origin")
    case (0, y):
        print(f"On Y axis at y={y}")
    case (x, 0):
        print(f"On X axis at x={x}")
    case (x, y):
        print(f"Point at ({x}, {y})")
