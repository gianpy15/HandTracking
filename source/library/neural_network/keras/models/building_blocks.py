import keras.layers as kl
import keras.models as km


def parallel_conv_block(in_tensor: km.Layer, end_ch: int, topology: tuple, kernel_size=(3, 3), activation='relu'):
    start_ch = in_tensor.output_shape[-1]

    outs = []
    for depth in topology:
        curr_ch = start_ch
        curr_tensor = in_tensor
        missing = depth
        # share ending channels between parallel ways
        target_end_ch = int(end_ch / len(topology))
        # to compensate imprecise int division give a little bonus here and there
        if len(outs) < end_ch - target_end_ch:
            target_end_ch += 1
        while missing > 0:
            ch_increment = int((target_end_ch - curr_ch) / missing)
            curr_ch += ch_increment
            curr_tensor = kl.Conv2D(filters=curr_ch, padding='same', kernel_size=kernel_size)(curr_tensor)
            curr_tensor = kl.Activation(activation)(curr_tensor) \
                if isinstance(activation, str) else activation()(curr_tensor)
            missing -= 1
        outs.append(curr_tensor)
    out = kl.concatenate(inputs=outs)
    assert out.output_shape[-1] == end_ch
    return out


def serial_pool_reduction_block(in_tensor: km.Layer, end_ch: int, topology: tuple, kernel_size=(3, 3), activation='relu'):
    start_ch = in_tensor.output_shape[-1]

    missing = sum(topology)

    curr_ch = start_ch
    curr_tensor = in_tensor
    for section in topology:
        for _ in range(section):
            ch_increment = int((end_ch - curr_ch) / missing)
            curr_ch += ch_increment
            curr_tensor = kl.Conv2D(filters=curr_ch, padding='same', kernel_size=kernel_size)(curr_tensor)
            curr_tensor = kl.Activation(activation)(curr_tensor) \
                if isinstance(activation, str) else activation()(curr_tensor)
            missing -= 1
        curr_tensor = kl.MaxPool2D(pool_size=(2, 2))(curr_tensor)

    return curr_tensor
