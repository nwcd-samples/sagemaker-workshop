import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="./")
    #不能使用 args = parser.parse_args()
    args, _ = parser.parse_known_args()
    files = os.listdir(args.path)
    print("=============list file begin")
    count = 0
    for file in files:
        print(file)
        count = count + 1
        if count == 100:
            break
    print("=============list file end")
