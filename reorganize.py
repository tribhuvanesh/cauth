"""
Re-organizes collected images from current directory and moves them into appropriate folders.
Also creates a text file which lists out names and location of images.
"""

import os, shutil
from collections import defaultdict
import json

EXTENSION = ".jpeg"

def main():
    # Get contents of the current working directory
    ls = os.listdir(os.getcwd())

    # Create "data" directory if it doesnt exist
    if "data" not in ls:
        os.mkdir( os.path.join(os.getcwd(), "data") )

    savePath = os.path.join(os.getcwd(), "data")

    user_map = defaultdict(int)
    user_index = 1
    lines = ""

    for file in ls:
        fileName, fileExt = os.path.splitext(file)

        src = os.path.join(os.getcwd(), file)

        if fileExt == EXTENSION:
            try:
                prefix, num = fileName.split('-')
            except ValueError:
                pass

            try:
                dst = os.path.join(savePath, prefix, file)
                shutil.move(src, dst)
                if prefix not in user_map.keys():
                    user_map[prefix] = user_index
                    user_index += 1
                lines += "%d %s\n" % (user_map[prefix], dst)

            except IOError:
                os.mkdir(os.path.join(savePath, prefix))

        with open( os.path.join(os.getcwd(), 'train.txt') , "w") as f:
            f.write(lines)

        str = json.dumps(user_map)
        with open(os.path.join(savePath, "user_map"), "w") as f:
            f.write(str)

if __name__ == '__main__':
    main()
