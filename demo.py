import tensorflow as tf
import numpy as np
import i2v.i2v as i2v
import i2v.ops as ops
import i2v.write_voxel as write_voxel
import PIL
slim = tf.contrib.slim

def demo(args):
    image = tf.placeholder(shape=[1, 127, 127, 3], dtype=tf.float32)
    predict = i2v.i2v(args,image)
    save = tf.train.Saver()
    with tf.Session() as sess:
        save.restore(sess, args.save_path)
        im = PIL.Image.open(args.image).resize((127,127))
        im_matrix = np.expand_dims(np.array(im)[:, :, 0:3].astype(np.uint8), axis=0)
        predict_ = sess.run(predict, feed_dict={image: im_matrix})
        predict_matrix = np.squeeze(predict_)
        write_voxel.voxel2obj(args.obj, predict_matrix > args.threshold)

if __name__ == '__main__':
    args = ops.parse_args()
    demo(args)