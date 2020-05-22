import tensorflow as tf
import numpy as np
import os,sys,argparse,math,time,glob,platform,i2v
import i2v.ops as ops
import i2v.i2v as i2v
from PIL import Image
from tflearn.layers.normalization import batch_normalization

slim = tf.contrib.slim
def pretrain(args):
    data_all = ops.get_data(args.image_dir,args.voxel_dir,0,1)
    valid_data_all = ops.get_data(args.image_dir,args.voxel_dir,0,1)
    batch_size = args.batch_size
    n_samples = len(data_all)
    total_batch = int(n_samples / batch_size)
    n_valid = len(valid_data_all)
    valid_batch = int(n_valid / batch_size)

    #model
    image_batch = tf.placeholder(shape=[batch_size,127, 127, 3],dtype=tf.float32)
    voxel_batch = tf.placeholder(shape=[batch_size, 32 ,32 , 32,1], dtype=tf.float32)
    predict = i2v.i2v(args,image_batch)
    acc = []
    for i in range(10):
        acc_f = ops.similar(voxel_batch,predict,args.threshold*(0.1+0.1 * i))
        acc.append(acc_f)
    weight = tf.placeholder(dtype=tf.float32)
    loss = tf.reduce_mean(
        ops.weighted_binary_crossentropy(tf.clip_by_value(tf.nn.sigmoid(predict), 1e-7, 1.0 - 1e-7),
                                     voxel_batch,weight))
    lr = tf.placeholder(dtype=tf.float32)
    train_op = tf.train.AdamOptimizer(lr).minimize(loss)
	
    save = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(args.epoch):
            np.random.shuffle(data_all)
            train_data_ = data_all
            for i in range(total_batch):
                offset = (i * batch_size) % (n_samples)
                batch_data = train_data_[offset:(offset + batch_size)]
                images = np.zeros([batch_size,127, 127, 3])
                voxels = np.zeros([batch_size, 32, 32, 32,1])
                for k in range(batch_size):
                    images[k,:,:,:] = batch_data[k,1]
                    voxels [k,:,:,:,:] = batch_data[k,0]
                sess.run(train_op, feed_dict={image_batch: images, voxel_batch: voxels, lr: args.lr, weight: args.l})
            if (epoch%args.show == 0)|(epoch == args.epoch-1):
                print("begin testing")
                a = np.zeros([10, valid_batch])
                test_begin = time.time()
                for i in range(valid_batch):
                    offset = (i * batch_size) % (n_valid)
                    batch_data = valid_data_all[offset:(offset + batch_size)]
                    images = np.zeros([batch_size,127, 127, 3])
                    voxels = np.zeros([batch_size, 32, 32, 32, 1])
                    for j in range(batch_size):
                        images[j, :, :, :] = batch_data[j, 1]
                        voxels[j, :, :, :, :] = batch_data[j, 0]
                    acc_ = sess.run(acc, feed_dict={image_batch: images, voxel_batch: voxels})
                    a[:, i] = acc_
                test_end = time.time()
                test_epoch_time = test_end - test_begin
                acc_mean = np.mean(a, axis=1)
                print("epoch "+str(epoch)+": test finished  " + "time_cost:" + str(test_epoch_time) +  "iou mean:" + str(acc_mean))
                if (epoch > 0)  |  (epoch == args.epoch - 1):
                    if np.max(acc_mean)>0.5:
                        save_path = "./models/" + "" + \
                                    str(int(100*np.max(test_epoch_time)*np.max(acc_mean)))+ ".ckpt"
                        save.save(sess, save_path)
        print("Traing finished")

if __name__ == '__main__':
    args = ops.parse_args()
    pretrain(args)