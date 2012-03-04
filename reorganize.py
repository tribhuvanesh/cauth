"""
Re-organizes collected images from current directory and moves them into appropriate folders.
Also creates a text file which lists out names and location of images.
"""

import os, shutil

EXTENSION = ".jpeg"

def main():
    # Get contents of the current working directory
    ls = os.listdir(os.getcwd())

    # Create "data" directory if it doesnt exist
    if "data" not in ls:
        os.mkdir( os.path.join(os.getcwd(), "data") )

    savePath = os.path.join(os.getcwd(), "data")

    for file in ls:
        fileName, fileExt = os.path.splitext(file)

        src = os.path.join(os.getcwd(), file)

        if fileExt == EXTENSION:
            prefix, num = fileName.split('-')

            if prefix == '' or num == '' : break

            try:
                dst = os.path.join(savePath, prefix)
                shutil.move(src, dst)
            except OSError:
                os.mkdir(os.path.join(savePath, prefix))


if __name__ == '__main__':
    main()
