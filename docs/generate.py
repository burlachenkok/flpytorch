#!/usr/bin/env python3

import glob, os, subprocess

# Create destantion folder for documentation
if not os.path.exists("./generated"): 
    os.makedirs("./generated")
gendocs_folder = os.path.abspath("generated")

print("Documentaiton will be generated in folder: ", gendocs_folder)

# Get path for template folder
template_dir = os.path.abspath("templates")

# Up to one folder
os.chdir("./../fl_pytorch")

# List of all python source files (modules and start scripts)
all_python_modules = [] 
for root, dirnames, filenames in os.walk('.'):
    for name in filenames:
        fullname = os.path.join(root, name)
        fname, ext = os.path.splitext(fullname)
        if ext == ".py" and fname.find("generated") == -1 and fullname.find("setup.py") == -1 and fullname.find("start.py") == -1:
            all_python_modules.append(os.path.abspath(fullname))

# Generate documentation for script and files
for module in all_python_modules:
    relpath = os.path.relpath(module, os.getcwd()).replace("\\", "/").replace("/", ".").replace(".py", "")
    if relpath.find("__init__") != -1:
        continue

    toRun = f"python -m pdoc --html --force --output-dir {gendocs_folder} --template-dir {template_dir} {relpath}"
    ret = subprocess.call(toRun, shell = True)
    if ret == 0:
        print("[OK]: ", toRun)
    else:
        print("[FAILED] return code: ", ret, " from command: ", toRun)
