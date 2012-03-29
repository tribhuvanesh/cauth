"""
Re-organizes collected images from current directory and moves them into appropriate folders.
Also creates a text file which lists out names and location of images.

Note:
ls : Current working directory
savePath : ./data/
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

    # Load the json user_map, if it exists
    try:
        user_map_data = open(os.path.join(savePath, 'user_map')).read()
        user_map = json.loads(user_map_data)
        user_index = max(user_map.values()) + 1
    except:
        # user_map does not exist. Create a new dic.
        user_map_flag = 1
        user_map = defaultdict(int)
        user_index = 1

    move_count = 0

    # Move all jpegs to data/_prefix/
    for file in ls:
        fileName, fileExt = os.path.splitext(file)

        src = os.path.join(os.getcwd(), file)

        if fileExt == EXTENSION:
            prefix, num = (None, None)
            try:
                prefix, num = fileName.split('-')
            except ValueError:
                pass

            if prefix and num:
                try:
                    dst = os.path.join(savePath, prefix, file)
                    shutil.move(src, dst)
                    if prefix not in user_map.keys():
                        user_map[prefix] = user_index
                        user_index += 1
                    # lines += "%d %s %s\n" % (user_map[prefix], prefix, dst)
                    move_count += 1
                except IOError:
                    os.mkdir(os.path.join(savePath, prefix))


    print "Moved %d files" % move_count
    print user_map

    # Iterate through all jpegs in data folder and list them out in train.txt
    img_path_lines = []
    for prefix in user_map.keys():
        cur_dir = os.path.join(savePath, prefix)
        ls_images = os.listdir(cur_dir)
        for img in ls_images:
            if os.path.splitext(img)[1] == EXTENSION:
                img_path_line =  "%d %s %s" % (user_map[prefix], prefix, os.path.join(savePath, prefix, img))
                # print img_path_line
                img_path_lines.append(img_path_line)

    with open("train.txt", "w") as f:
        # lines = sorted(img_path_lines, key=lambda x:int(x.split()[0]))
        lines = sorted(img_path_lines)
        for line in lines : f.write(line+"\n")

    # Dump current contents of user_map into file
    with open(os.path.join(savePath, 'user_map'), "w") as f:
        f.write(json.dumps(user_map))


if __name__ == '__main__':
    main()
