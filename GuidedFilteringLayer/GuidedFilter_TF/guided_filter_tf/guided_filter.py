import tensorflow as tf

from .box_filter import box_filter


def guided_filter(x, y, r, eps=1e-8, nhwc=False):
    assert x.shape.ndims == 4 and y.shape.ndims == 4

    # data format
    if nhwc:
        x = tf.transpose(x, [0, 3, 1, 2])
        y = tf.transpose(y, [0, 3, 1, 2])

    # shape check
    x_shape = tf.shape(x)
    y_shape = tf.shape(y)

    assets = [tf.assert_equal(   x_shape[0],  y_shape[0]),
              tf.assert_equal(  x_shape[2:], y_shape[2:]),
              tf.assert_greater(x_shape[2:],   2 * r + 1),
              tf.Assert(tf.logical_or(tf.equal(x_shape[1], 1),
                                      tf.equal(x_shape[1], y_shape[1])), [x_shape, y_shape])]

    with tf.control_dependencies(assets):
        x = tf.identity(x)

    # N
    N = box_filter(tf.ones((1, 1, x_shape[2], x_shape[3]), dtype=x.dtype), r)

    # mean_x
    mean_x = box_filter(x, r) / N
    # mean_y
    mean_y = box_filter(y, r) / N
    # cov_xy
    cov_xy = box_filter(x * y, r) / N - mean_x * mean_y
    # var_x
    var_x  = box_filter(x * x, r) / N - mean_x * mean_x

    # A
    A = cov_xy / (var_x + eps)
    # b
    b = mean_y - A * mean_x

    mean_A = box_filter(A, r) / N
    mean_b = box_filter(b, r) / N

    output = mean_A * x + mean_b

    if nhwc:
        output = tf.transpose(output, [0, 2, 3, 1])

    return output


def fast_guided_filter(lr_x, lr_y, hr_x, r, eps=1e-8, nhwc=False):
    assert lr_x.shape.ndims == 4 and lr_y.shape.ndims == 4 and hr_x.shape.ndims == 4

    # data format
    if nhwc:
        lr_x = tf.transpose(lr_x, [0, 3, 1, 2])
        lr_y = tf.transpose(lr_y, [0, 3, 1, 2])
        hr_x = tf.transpose(hr_x, [0, 3, 1, 2])

    # shape check
    lr_x_shape = tf.shape(lr_x)
    lr_y_shape = tf.shape(lr_y)
    hr_x_shape = tf.shape(hr_x)

    assets = [tf.assert_equal(   lr_x_shape[0], lr_y_shape[0]),
              tf.assert_equal(   lr_x_shape[0], hr_x_shape[0]),
              tf.assert_equal(   lr_x_shape[1], hr_x_shape[1]),
              tf.assert_equal(  lr_x_shape[2:], lr_y_shape[2:]),
              tf.assert_greater(lr_x_shape[2:], 2 * r + 1),
              tf.Assert(tf.logical_or(tf.equal(lr_x_shape[1], 1),
                                      tf.equal(lr_x_shape[1], lr_y_shape[1])), [lr_x_shape, lr_y_shape])]

    with tf.control_dependencies(assets):
        lr_x = tf.identity(lr_x)

    # N
    N = box_filter(tf.ones((1, 1, lr_x_shape[2], lr_x_shape[3]), dtype=lr_x.dtype), r)

    # mean_x
    mean_x = box_filter(lr_x, r) / N
    # mean_y
    mean_y = box_filter(lr_y, r) / N
    # cov_xy
    cov_xy = box_filter(lr_x * lr_y, r) / N - mean_x * mean_y
    # var_x
    var_x  = box_filter(lr_x * lr_x, r) / N - mean_x * mean_x

    # A
    A = cov_xy / (var_x + eps)
    # b
    b = mean_y - A * mean_x

    # mean_A; mean_b
    A    = tf.transpose(A,    [0, 2, 3, 1])
    b    = tf.transpose(b,    [0, 2, 3, 1])
    hr_x = tf.transpose(hr_x, [0, 2, 3, 1])

    mean_A = tf.image.resize_images(A, hr_x_shape[2:])
    mean_b = tf.image.resize_images(b, hr_x_shape[2:])

    output = mean_A * hr_x + mean_b

    if not nhwc:
        output = tf.transpose(output, [0, 3, 1, 2])

    return output