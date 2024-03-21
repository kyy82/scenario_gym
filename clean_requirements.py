import re

# Path to the original and new requirements file
original_requirements_path = 'requirements.txt'
clean_requirements_path = 'clean_requirements.txt'

# Regular expression to match package and version
package_version_re = re.compile(r'^([a-zA-Z0-9_-]+)==([0-9\.]+)$')
package_name_re = re.compile(r'^([a-zA-Z0-9_-]+) @ ')

with open(original_requirements_path, 'r') as file:
    lines = file.readlines()

with open(clean_requirements_path, 'w') as file:
    for line in lines:
        # Check for standard package==version format
        match = package_version_re.match(line)
        if match:
            file.write(line)
        else:
            # Attempt to extract package name from other formats
            name_match = package_name_re.search(line)
            if name_match:
                file.write(f"{name_match.group(1)}\n")
            else:
                # If it's a git repository or other non-standard format, handle separately
                if 'git+' in line:
                    # Extract package name from git URL (you might need to adjust this)
                    package_name = line.split('#egg=')[-1].strip()
                    file.write(f"{package_name}\n")
                else:
                    print(f"Skipping unrecognized format: {line.strip()}")

print(f"Clean requirements written to {clean_requirements_path}")
