import tensorflow as tf

def filter_image(origin_image, kernel):
    image = tf.expand_dims(origin_image, 0)
    kernel = tf.expand_dims(kernel, -1) 
    output = tf.nn.depthwise_conv2d(\
        image, kernel, strides=(1, 1, 1, 1), padding="VALID")
    output = tf.squeeze(output)
    return output

def gaussian_filtered_image(image, sensor):
    if sensor == "WV2":
        gauss_kernel = hWV2
    elif sensor == "WV3":
        gauss_kernel = hWV3
    elif sensor == "GeoEye1":
        gauss_kernel = hGE1
    else:
        gauss_kernel = hIK
        
    return  filter_image(image, gauss_kernel)

def d_s_reg(ms, pan):
  ms = tf.reshape(ms, (ms.shape[0] * ms.shape[1], ms.shape[2]))
  pan = tf.reshape(pan, (pan.shape[0] * pan.shape[1],1))

  alpha = tf.matmul(pinv(ms), pan)
  fi = tf.matmul(ms, alpha)
  return 1 - r_squared(pan, fi)


def d_lambda(ms, fused, p, b, sensor):
  fused_filtered = gaussian_filtered_image(fused, sensor)
  return 1 - q_index(fused_filtered, ms[R:-R, R:-R, :])

def hqnr(f, ms, pan, alpha, beta, bands, sensor):
  a = ( 1 - d_lambda(\
      ms, f, b=bands, sensor=sensor))**alpha
  b = ( 1 - d_s_reg(f, pan)) ** beta
  return a*b