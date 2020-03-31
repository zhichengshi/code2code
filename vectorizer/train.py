import os
import _pickle as pkl
import tensorflow as tf
import word2vec as network
from config import NUM_FEATURES, BATCH_SIZE, EPOCHS, CHECKPOINT_EVERY, initial_learning_rate, decay_rate, decay_steps
from tensorflow.contrib.tensorboard.plugins import projector
from tqdm import tqdm
'''
samples:批处理数据
logdir:checkpoint
outfile:embedding
'''



# 获得批处理数据
def batchSamples(samples, batchSize):
    batch = ([], [])
    count = 0
    # indexOf = lambda x: nodeMap[x]
    for sample in tqdm(samples):
        label = sample[0]
        tag=sample[1]

        batch[0].append(label)
        batch[1].append(tag)
        count += 1
        if count >= batchSize:
            yield batch
            batch, count = ([], []), 0

def learn_vectors(samples, logdir, outfile,vocab_size, num_feats=NUM_FEATURES, epochs=EPOCHS):

    # 构建网络
    # num_feats:输出节点的维度
    input_node, label_node, embed_node, loss_node = network.init_net(
        vocabulary_size=vocab_size,
        num_feats=num_feats,
        batch_size=BATCH_SIZE
    )

    global_ = tf.Variable(tf.constant(0))
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_, decay_steps, decay_rate, staircase=True)

    # use gradient descent with momentum to minimize the training objective
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_node)

    tf.summary.scalar('loss', loss_node)

    # init the graph
    sess = tf.Session()

    with tf.name_scope('saver'):
        saver = tf.train.Saver()
        summaries = tf.summary.merge_all()

        writer = tf.summary.FileWriter(logdir, sess.graph)
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = embed_node.name
        projector.visualize_embeddings(writer, config)

    sess.run(tf.global_variables_initializer())

    checkfile = os.path.join(logdir, 'ast2vec.ckpt')

    embed_file = open(outfile, 'wb')

    step = 1


# 训练模型
    for epoch in range(1, epochs + 1):
        sample_gen = batchSamples(samples, BATCH_SIZE)
        for batch in sample_gen:
            input_batch, label_batch = batch

            _, summary, embed, err = sess.run(
                [train_step, summaries, embed_node, loss_node],
                feed_dict={
                    input_node: input_batch,
                    label_node: label_batch,
                }
            )

            if step % 10000 == 0:
                print('Epoch: ', epoch, 'Loss: ', err, 'step: ', step)
            writer.add_summary(summary, step)
            if step % CHECKPOINT_EVERY == 0:
                # save state so we can resume later
                saver.save(sess, os.path.join(checkfile), step)
                # save embeddings
                pkl.dump((embed, dictionary), embed_file)
            step += 1

    # save embeddings and the mapping
    pkl.dump((embed, dictionary), embed_file)
    embed_file.close()
    saver.save(sess, os.path.join(checkfile), step)


if __name__ == "__main__":
    sample_path = "dataset/context.pkl"
    out_file_path = "dataset/embeddings.pkl"
    dict_path='dataset/dict.pkl'
    
    log_dir = "log/word2vec"
    with open(sample_path, "rb") as f:
        samples = pkl.load(f)
    
    with open(dict_path,'rb') as f:
        dictionary=pkl.load(f)
    
    vocab_size=len(dict_path)+1 

    print("start to train word2vec")
    learn_vectors(samples, log_dir, out_file_path,vocab_size)
