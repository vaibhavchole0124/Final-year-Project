
import os

file_path = '/Users/vaibhavgovindchole/Employee_Attrition_AI_Project/app/templates/admin/admin_dashboard.html'

with open(file_path, 'r') as f:
    content = f.read()

# The specific broken pattern found via cat
broken_pattern = """            data: {
              labels: {{ month_labels | tojson }
      },
        datasets: [{"""

# The correct pattern
fixed_pattern = """            data: {
              labels: {{ month_labels | tojson }},
              datasets: [{"""

if broken_pattern in content:
    print("Found broken pattern! Fixing...")
    new_content = content.replace(broken_pattern, fixed_pattern)
    with open(file_path, 'w') as f:
        f.write(new_content)
    print("Fixed.")
else:
    print("Broken pattern not found exactly. Searching for partial...")
    # Fallback: try to find just the line
    if "labels: {{ month_labels | tojson }\n" in content:
        print("Found the bad line at least.")
        # This is the same logic as before, but let's be careful
        # We'll just replace that specific line and the next few lines manually if needed
        # But really, if the pattern didn't match, whitespace might be issue.
    else:
        print("Could not find the broken code.")
