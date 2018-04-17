from hand_data_management.frame_caching import *

FLAG_PROCESSING = '6'
FLAG_LABELED = '0'
FLAG_UNLABELED = '1'


def get_index_content(vidname):
    index_complete_path = get_index_from_vidname(vidname)
    index = open(index_complete_path, "r")
    content = index.read()
    index.close()
    return content


def tick_index_counters(vidname):
    index_path = get_index_from_vidname(vidname)
    index = open(index_path, "r+")
    content = index.read()
    updated_content = ''
    for frameno in range(len(content)):
        framecode = content[frameno]
        if framecode == FLAG_UNLABELED or framecode == FLAG_LABELED:
            updated_content += framecode
        else:
            newval = str(int(framecode) - 1)
            updated_content += newval
            if newval == FLAG_UNLABELED:
                uncache_frame(frame_name(vidname, frameno))
    index.seek(0)
    index.write(updated_content)
    index.close()


def build_empty_index_file(complete_filename, index_len):
    f = open(complete_filename, "x")
    f.write(FLAG_UNLABELED * index_len)
    f.close()


def set_index_flag(complete_filename, flag, idx):
    f = open(complete_filename, "r+")
    f.seek(idx)
    f.write(flag)
    f.close()
