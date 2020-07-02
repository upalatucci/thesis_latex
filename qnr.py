import tensorflow as tf
import tensorflow_probability as tfp

def q_index(y_true, y_pred):
    two = tf.constant(2.0, tf.float32)
    
    cov_b = tfp.stats.covariance(y_true,y_pred, [0, 1], None)
    true_b_std = tf.math.reduce_std(y_true, [0, 1])
    pred_b_std = tf.math.reduce_std(y_pred, [0, 1])

    true_b_mean = tf.cast(tf.reduce_mean(y_true, [0, 1]), tf.float32)
    pred_b_mean = tf.cast(tf.reduce_mean(y_pred, [0, 1]), tf.float32)
    
    q1_b = cov_b / (true_b_std * pred_b_std)
    q2_b = (two * true_b_mean * pred_b_mean) / (tf.square(true_b_mean) + tf.square(pred_b_mean))
    q3_b = (two * true_b_std * pred_b_std ) / (tf.square(true_b_std) + tf.square(pred_b_std))

    q_b = q1_b * q2_b * q3_b
    return tf.reduce_mean(q_b)


def d_lambda(ms, fused, p, b): 
    result = tf.constant(0.0, tf.float32)
    for l in range(b-1):
        for r in range(l+1, b):
            result += tf.abs(tf.cast(q_index(fused[:, :, l:l+1], fused[:, :, r:r+1]), tf.float32) - \
            tf.cast(q_index(ms[:, :, l:l+1], ms[:, :, r:r+1]), tf.float32))**p

    b = tf.constant(b, tf.float32)

    s = ( b * ( b - tf.constant(1.0, tf.float32) ) ) / tf.constant(2.0, tf.float32)
    
    result = result / s
    result = result ** (1.0/p)
    return result


def d_s(ms, fused, pan, pan_degraded, q, b):
    result = tf.constant(0.0, tf.float32)
    
    for l in range(b):
        result += tf.abs(tf.cast(q_index(fused[:, :, l:l+1], pan), tf.float32) - \
         tf.cast(q_index(ms[:, :, l:l+1], pan_degraded), tf.float32))**q

    b = tf.constant(b, tf.float32)

    result = result / b
    
    r = result**(1./q)
    return r


def qnr(fused, ms, pan, pan_degraded, alpha, beta, p, q, bands):
    a = (1-d_lambda(ms, fused, p=p, b=bands))**alpha
    b = (1-d_s(ms, fused, pan, pan_degraded, q, b=bands))**beta
    return a*b
