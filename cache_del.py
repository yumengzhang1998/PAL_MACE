import os
import shutil

def delete_pycaches(root_dir='.'):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for dirname in dirnames:
            if dirname == '__pycache__':
                full_path = os.path.join(dirpath, dirname)
                print(f"Deleting {full_path}")
                shutil.rmtree(full_path)

if __name__ == "__main__":
    delete_pycaches()