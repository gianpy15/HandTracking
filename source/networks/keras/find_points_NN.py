from keras.models import Sequential
import keras.layers as kl
from keras.initializers import RandomNormal
from keras.optimizers import Adam, SGD
from keras.losses import mean_squared_error
import numpy as np
import junctions_locator_utils.junction_locator_ds_management as jlocator
import hands_regularizer.regularizer as regularizer
import hands_bounding_utils.utils as utils

# some values

resize = 200
hm_resize = 100
input_height = resize
input_width = resize
threshold = .5
batch_size = 25
num_filters = 15
kernel = [4, 4]
pool = (2, 2)

# input building

img_reg = regularizer.Regularizer()
img_reg.fixresize(resize, resize)
hm_reg = regularizer.Regularizer()
hm_reg.fixresize(hm_resize, hm_resize)
hm_reg.heatmaps_threshold(threshold)
jlocator.create_dataset(["handsAlberto1"], im_regularizer=img_reg, heat_regularizer=hm_reg, enlarge=.5, cross_radius=5)
cuts, hms, visible = jlocator.read_dataset(verbosity=1)
utils.showimage(cuts[1])
utils.showimages(hms[1])
print(visible[1])

x_train = np.array(cuts)
print(x_train.shape)
y_train = np.array(hms)

# model building
model = Sequential()

model.add(kl.Conv2D(filters=num_filters,
                    kernel_size=kernel,
                    kernel_initializer=RandomNormal,
                    activation='relu',
                    input_shape=(None, None, 3),
                    data_format="channels_last",
                    padding="same"
                    )
          )

model.add(kl.Conv2D(filters=num_filters,
                    kernel_size=kernel,
                    kernel_initializer=RandomNormal,
                    activation='relu',
                    # input_shape=(batch_size, input_height, input_width, 3),
                    data_format="channels_last"
                    )
          )

model.add(kl.Conv2D(filters=num_filters,
                    kernel_size=kernel,
                    kernel_initializer=RandomNormal,
                    activation='relu',
                    # input_shape=(batch_size, input_height, input_width, 3),
                    data_format="channels_last"
                    )
          )
model.add(kl.Conv2D(filters=num_filters,
                    kernel_size=kernel,
                    kernel_initializer=RandomNormal,
                    activation='relu',
                    # input_shape=(batch_size, input_height, input_width, 3),
                    data_format="channels_last"
                    )
          )

model.add(kl.Conv2D(filters=num_filters,
                    kernel_size=kernel,
                    kernel_initializer=RandomNormal,
                    activation='relu',
                    # input_shape=(batch_size, input_height, input_width, 3),
                    data_format="channels_last"
                    )
          )

model.add(kl.Conv2D(filters=num_filters,
                    kernel_size=kernel,
                    kernel_initializer=RandomNormal,
                    activation='relu',
                    # input_shape=(batch_size, input_height, input_width, 3),
                    data_format="channels_last"
                    )
          )

model.add(kl.Conv2D(filters=num_filters,
                    kernel_size=kernel,
                    kernel_initializer=RandomNormal,
                    activation='relu',
                    # input_shape=(batch_size, input_height, input_width, 3),
                    data_format="channels_last"
                    )
          )

model.add(kl.Conv2D(filters=num_filters,
                    kernel_size=kernel,
                    kernel_initializer=RandomNormal,
                    activation='relu',
                    # input_shape=(batch_size, input_height, input_width, 3),
                    data_format="channels_last"
                    )
          )

model.add(kl.Conv2D(filters=num_filters,
                    kernel_size=kernel,
                    kernel_initializer=RandomNormal,
                    activation='relu',
                    # input_shape=(batch_size, input_height, input_width, 3),
                    data_format="channels_last"
                    )
          )

model.add(kl.Conv2D(filters=num_filters,
                    kernel_size=kernel,
                    kernel_initializer=RandomNormal,
                    activation='relu',
                    # input_shape=(batch_size, input_height, input_width, 3),
                    data_format="channels_last"
                    )
          )

model.add(kl.Conv2D(filters=num_filters,
                    kernel_size=kernel,
                    kernel_initializer=RandomNormal,
                    activation='relu',
                    # input_shape=(batch_size, input_height, input_width, 3),
                    data_format="channels_last"
                    )
          )

model.add(kl.Conv2D(filters=num_filters,
                    kernel_size=kernel,
                    kernel_initializer=RandomNormal,
                    activation='relu',
                    # input_shape=(batch_size, input_height, input_width, 3),
                    data_format="channels_last"
                    )
          )

# model.add(kl.MaxPooling2D(pool_size=pool))

# config

model.compile(
    optimizer=Adam(),
    loss=mean_squared_error(),
    metrics=['accuracy']
)

# training

model.fit(epochs=100, batch_size=batch_size, x=x_train, y=y_train)
