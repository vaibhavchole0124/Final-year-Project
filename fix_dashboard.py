
import os

file_path = '/Users/vaibhavgovindchole/Employee_Attrition_AI_Project/app/templates/admin/admin_dashboard.html'

with open(file_path, 'r') as f:
    content = f.read()

# The target bad string (flexible with whitespace)
bad_string_part = "labels: {{ month_labels | tojson }\n      },\n        datasets: [{"

# Alternative bad string if indentation is different
bad_string_part_2 = "labels: {{ month_labels | tojson }\n"

print(f"Current length: {len(content)}")

if "labels: {{ month_labels | tojson }\n" in content:
    print("Found the bad pattern!")
    
    # We will replace the specific block more broadly to be safe
    # Locate the block start
    start_marker = "if (monthCtx) {"
    end_marker = "options: {"
    
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        print(f"Replacing block between indices {start_idx} and {end_idx}")
        
        pre_block = content[:start_idx]
        post_block = content[end_idx:]
        
        new_block = """if (monthCtx) {
          new Chart(monthCtx, {
            type: 'bar',
            data: {
              labels: {{ month_labels | tojson }},
              datasets: [{
                label: 'Not At Risk',
                data: {{ safe_month | tojson }},
                backgroundColor: '#3b82f6'
              }, {
                label: 'At Risk',
                data: {{ at_risk_month | tojson }},
                backgroundColor: '#facc15'
              }]
            },
            """
            
        new_content = pre_block + new_block + post_block
        
        with open(file_path, 'w') as f:
            f.write(new_content)
        print("File updated successfully!")
    else:
        print("Could not find start/end markers for the block.")
else:
    print("Could not find the specific bad pattern, printing snippet around line 454:")
    lines = content.splitlines()
    if len(lines) > 450:
        for i in range(450, min(460, len(lines))):
            print(f"{i+1}: {lines[i]}")

