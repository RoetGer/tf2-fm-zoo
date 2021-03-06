import glob
import importlib
import os.path as osp


## Import all the modules in the directory

glob_str = osp.join(osp.dirname(__file__), "*.py")
modules = glob.glob(glob_str)

# Clean up module names
modules = [ 
    osp.basename(f).replace(".py", "") 
        for f in modules if osp.isfile(f) and not f.endswith("__init__.py")
]

for module_name in modules:
    importlib.import_module(f"fm_zoo.{module_name}")