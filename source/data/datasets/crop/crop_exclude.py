from data import *

def multiple_hands_video_list():
    path = resources_path("framedata", "cropExcluded.txt")
    with open(path, "r") as f:
        vidlist = [vid.strip('\n') for vid in f]

    return vidlist


if __name__ == '__main__':
    print(multiple_hands_video_list())