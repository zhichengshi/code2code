from single_astnn.config import *
import sys
from single_astnn import model
import tensorflow as tf
from sampling import *
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def test_model(logdir, pickle_data_path, vector, vector_lookup):
    # init the network
    nodes_node, children_node, statement_len_list, code_vector, logits = model.init_net(
        Node_Embedding_Size, label_size
    )

    # for calculate the training accuracy
    out_node = model.out_layer(logits)

    sess = tf.Session()

    with tf.name_scope('saver'):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(logdir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,logdir+"/cnn_tree.ckpt-380000")
        else:
            raise ('Checkpoint not found.')

    correct_labels = []
    # make predictions from the input
    predictions = []
    step = 1
    with open(pickle_data_path,"rb") as f:
        elements=pickle.load(f)

    elements=elements[:1000]
    for nodes, children, statement_len, label_vector in generate_sample_from_list(elements, vector,
                                                                                   vector_lookup):
        output = sess.run([out_node], feed_dict={
            nodes_node: nodes,
            children_node: children,
            statement_len_list: statement_len
        })

        correct_labels.append(np.argmax(label_vector))
        predictions.append(np.argmax(output))
        step += 1

        if step % 100 == 0:
            print(step)

    labels = []
    for i in range(1, label_size + 1):
        labels.append(str(i))

    print('Accuracy:', accuracy_score(correct_labels, predictions))
    print(classification_report(correct_labels, predictions, target_names=labels))

    # prediction_list = list(set(predictions))
    # prediction_list.sort()
    #
    # print("预测值：")
    # print(prediction_list)
    # print(len(set(predictions)))
    #
    #
    # correct_labels_list=list(set(correct_labels))
    # correct_labels_list.sort()
    #
    # print("真實值")
    # print(correct_labels_list)
    # print(len(set(correct_labels_list)))


    # print(confusion_matrix(correct_labels, predictions))


if __name__ == "__main__":
    sys.setrecursionlimit(1000000)
    # node_embedding_path="/home/cheng/桌面/matchFunciton/log/104_label/word2vec_log/embeddingFile.data"
    # with open(node_embedding_path, "rb") as f:
    #     data = pickle.load(f)
    #     vector = data[0]
    #     vector_lookup = data[1]
    #
    # test_path = "/home/cheng/桌面/matchFunciton/data/label_code/104_label_code/validate.pkl"
    # sess_graph_path="/home/cheng/桌面/matchFunciton/log/104_label/rnn_ast_log"sd
    # test_model(sess_graph_path,test_path, vector, vector_lookup)

    node_embedding_path = "/home/cheng/Desktop/algorithm_classify/data/embeddings/non_leaf_feature_40.data"
    with open(node_embedding_path, "rb") as f:
        data = pickle.load(f)
        vector = data[0]
        vector_lookup = data[1]

    # test_path = "/home/cheng/桌面/matchFunciton/data/label_code/104_label_code/test1.pkl"
    test_path = "/home/cheng/Desktop/algorithm_classify/data/bubble_detection/dataset/train.pkl"
    sess_graph_path = "/home/cheng/Desktop/algorithm_classify/log/bubble_sort_detection/log_astnn"
    test_model(sess_graph_path, test_path, vector, vector_lookup)
