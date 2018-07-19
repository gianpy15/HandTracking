import os
from data.naming import *
from library.neural_network import heatmap
from runnables.evaluation.eval_functions import *
from data.datasets.crop.hands_locator_from_rgbd import create_dataset_shaded_heatmaps as cropscreate, read_dataset
from data.datasets.crop.jsonhands_dataset_manager import create_dataset_shaded_heatmaps_synth as jsonscreate, read_dataset as jsonread
from skimage.transform import resize
import tqdm


# ###### TOUCH ########
def model():
    return heatmap()


# TEST DATASET PATH
test_ds_path = jsonhands_path()

# NEED TO BUILD? IF NO, COMMENT. I NEEDED IT WHEN I DID THIS
# cropscreate(savepath=test_ds_path, fillgaps=False, resize_rate=0.5, width_shrink_rate=4, heigth_shrink_rate=4)
# jsonscreate(dspath=resources_path(os.path.join("hand_labels_synth", "synth2")), savepath=test_ds_path, resize_rate=0.5, width_shrink_rate=4, heigth_shrink_rate=4)

# AFTER THIS, x AND y HAVETO BE THE INPUT AND EXPECTED OUTPUT OF THE NET
x, y = jsonread(test_ds_path)

# EVENTUAL PREPROCESS FUNs. THIS  WILL BE APPLIED TO EACH SAMPLE BEFORE BEING FED TO THE NETWORK
def preprocess_x(samp):
    samp = resize(samp, output_shape=(224, 224, 3))
    return np.expand_dims(samp, axis=0)


def preprocess_y(samp):
    samp = samp.squeeze()
    samp *= 255
    samp = resize(samp, output_shape=(56, 56))
    samp /= 255
    return samp


# THOSE HAVE TO BE FUNCTIONS THAT TAKE AS INPUT TWO ARRAYS: Y_TRUE, Y_PRED (IN THIS ORDER!!)
# Y_TRUE AND Y_PRED ARE RESPECTIVELY ALL THE EXPECTED OUTPUTS AND ACTUAL OUTPUTS OF THE NETWORK
# NOTHING IS DONE WITH EVENTUAL RETURN VALUES. IF YOU HAVE SOMETHING TO PRINT, PRINT IT IN THE FUNCTION, OKAY?
analysis = [loop_pix_avg_dist, loop_prec_recall]


# ###### DONT TOUCH ######
net = model()

yp = []
yt = []
xt = []
for i in tqdm.tqdm(range(len(x))):
    sample = x[i]
    yt.append(preprocess_y(y[i]))
    sample = preprocess_x(sample)
    xt.append(sample[0])
    yp.append(net.predict(sample).squeeze())

for eval_fun in analysis:
    eval_fun(yt, yp)


'''
def convert_heats(heats):
    ris = []
    for h in tqdm.tqdm(heats):
        ch = np.zeros([224, 224])
        for i1 in range(224):
            for j in range(224):
                ch[i1][j] = h[int(i1/4)][int(j/4)]
        ris.append(ch)
    return ris


xt = np.array(xt) * 255
yp = np.array(convert_heats(yp))
yp = 1 - (1 - yp)**10

create_sprite(25, xt, yp, num=2)



'''