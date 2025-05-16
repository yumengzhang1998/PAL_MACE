import os
import shutil


def delete_files_and_directories():
    if os.path.exists('results/TestRun'):
        shutil.rmtree('results/TestRun')
    if os.path.exists('logs'):
        shutil.rmtree('logs')
    if os.path.exists('results/xtbfail'):
        os.remove('results/xtbfail')

if __name__ == "__main__":

    # Your main script logic here
    # After the main logic, clean up the files and directories
    delete_files_and_directories()
    print("Files and directories deleted.")