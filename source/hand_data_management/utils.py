import skvideo.io as skio
from tqdm import tqdm

from hand_data_management.index import *
from image_loader.hand_io import *
from hand_data_management.frame_caching import *


def build_frame_root_from_vid(videopath, post_process=lambda f: None):
    videopath = pm.resources_path(videopath)
    videoname = os.path.splitext(os.path.split(videopath)[1])[0]
    framesdir = get_vid_dir_from_vidname(videoname)
    if os.path.exists(framesdir):
        return False

    videodata = skio.vread(pm.resources_path(videopath))
    os.makedirs(framesdir)
    post_process(framesdir)
    tmp = os.path.join(framesdir, TEMPDIR)
    os.makedirs(tmp)
    post_process(tmp)
    for frameidx in tqdm(range(len(videodata))):
        framefile = os.path.join(framesdir, frame_name(videoname, frameidx))
        store(framefile, videodata[frameidx])
        post_process(framefile)
    idxfile = os.path.join(framesdir, index_name(videoname))
    build_empty_index_file(idxfile, len(videodata))
    post_process(idxfile)
    return True


def save_labels(labels, frame):
    vidn = get_vidname(frame)
    framedir = os.path.join(framebase, vidn)
    frame = get_complete_frame_path(frame_name(vidn, get_frameno(frame)))
    data, _ = load(frame)
    store(frame, data=data, labels=labels)
    set_index_flag(os.path.join(framedir, index_name(vidn)),
                   flag=FLAG_LABELED,
                   idx=get_frameno(frame))


def select_best_frame(vidname):
    viddir = os.path.join(framebase, vidname)
    index = open(os.path.join(viddir, index_name(vidname)), "r")
    flagset = index.read()
    index.close()

    # give precedence to first and last frame for interpolation
    if flagset[0] == FLAG_UNLABELED:
        return 0, len(flagset)
    elif flagset[-1] == FLAG_UNLABELED:
        return len(flagset)-1, len(flagset)

    # or find the frame that minimizes the largest unlabeled interval
    best_start = flagset.find(FLAG_UNLABELED)
    if best_start == -1:
        return -1, -1
    best_end = best_start
    current_start = best_start
    current_end = best_start + 1
    while current_end < len(flagset):
        if flagset[current_end] != FLAG_UNLABELED:
            if current_end - current_start - 1 > best_end - best_start:
                best_end = current_end - 1
                best_start = current_start
            while current_end < len(flagset) and flagset[current_end] != FLAG_UNLABELED:
                current_end += 1
            current_start = current_end
        current_end += 1
    if current_end - current_start - 1 > best_end - best_start:
        best_end = current_end - 1
        best_start = current_start
    return (best_end + best_start) // 2, best_end - best_start + 1


def select_best_overall_frame():
    vids = os.listdir(framebase)
    bestvid = ('', -1, -1)
    vids = [os.path.join(framebase, vid) for vid in vids]
    vids = [vid_dir for vid_dir in vids if os.path.isdir(vid_dir)]
    vids = [vid.split('/')[-1] for vid in vids]
    for vidname in vids:
        selected_frame, frames_interval = select_best_frame(vidname)
        if frames_interval > bestvid[2]:
            bestvid = (vidname, selected_frame, frames_interval)
    if bestvid[2] == -1:
        return None
    selected_vid_dir = os.path.join(framebase, bestvid[0])
    selected_frame_name = os.path.join(selected_vid_dir,
                                       frame_name(bestvid[0],
                                                  bestvid[1]))
    fdata, _ = load(selected_frame_name)
    return fdata, frame_name(bestvid[0],
                             bestvid[1])


def add_contributor(nick):
    contribs = open(contributors, "r+")
    c = contribs.readline()
    pos = len(c)
    while c != '' and c.split(' ')[0] != nick:
        c = contribs.readline()
        pos += len(c)

    if c == '':
        contribs.write(nick + " 0001\n")
    else:
        current_amount = int(c.split(' ')[1])
        contribs.seek(pos - len(c))
        contribs.write(nick + " %04d" % (current_amount + 1,))


def register_labels(labelstring, frame, contributor=None):
    tokens = labelstring.split(',')
    if len(tokens) != 63:
        return False
    raw_labels = []
    for idx in range(21):
        raw_labels.append((float(tokens[3*idx]),
                           float(tokens[3*idx+1]),
                           1 if tokens[3*idx+2] in ('true', 'True', 'TRUE') else 0))
    save_labels(labels=raw_labels, frame=frame)
    if contributor is not None:
        add_contributor(contributor.replace(" ", ""))
    else:
        add_contributor("Anonymous")
    uncache_frame(frame)
    tick_index_counters(get_vidname(frame))
    return True
