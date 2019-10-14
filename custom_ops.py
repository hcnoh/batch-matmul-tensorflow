import tensorflow as tf


def tf_batch_matmul(a, b):
    """

    Args:
        a, b: '3-D' tensors with shape '[None, num_rows, num_cols]'.

    Returns:
        The result of batch matrix multiplication of two batched matrixes a, b.
    
    """
    a_row, a_col = a.shape[1], a.shape[2]
    b_row, b_col = b.shape[1], b.shape[2]
    
    tiled_a = tf.reshape(tf.tile(a, [1, b_col, 1]), shape=[-1, b_col, a_row, a_col])
    tiled_b = tf.reshape(tf.tile(b, [1, 1, a_row]), shape=[-1, b_row, a_row, b_col])
    
    return tf.reduce_sum(
        tf.transpose(tiled_a, [0, 2, 3, 1]) * tf.transpose(tiled_b, [0, 2, 1, 3]), axis=2
    )