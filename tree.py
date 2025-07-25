import os

def print_tree(start_path, prefix=""):
    for item in sorted(os.listdir(start_path)):
        path = os.path.join(start_path, item)
        if os.path.isdir(path):
            print(prefix + "📁 " + item)
            print_tree(path, prefix + "    ")
        else:
            print(prefix + "📄 " + item)

print_tree(".")
