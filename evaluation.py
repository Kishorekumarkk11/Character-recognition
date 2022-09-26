# Programmers : Dev Bishnoi

# Evaluation on testing data
import tensorflow as tf
import csv
import feed_data
import os
import tensorflow.compat.v1 as tf1
batch_size = 40
file_handle = open("C:\\Users\\DELL\\PycharmProjects\\Ml lab 4\\emnist-test.csv", "r")
reader = csv.reader(file_handle, delimiter = ',')

def getLength():
	cnt = 0
	for row in reader:
		cnt += 1
	return cnt

def set_file_handle_to_beggining():
	file_handle.seek(0)
	return

def evaluation():
	pretrained = os.path.isdir('store')
	if(pretrained == False):
		print("Please train your model first")
		return -1

	test_size = getLength()
	set_file_handle_to_beggining()

	with tf.compat.v1.Session() as sess:
		saver = tf1.train.import_meta_graph("store/my_model.meta")
		saver.restore(sess, tf.train.latest_checkpoint("store/"))
		input_x = tf1.get_collection("input_x")[0]
		input_y = tf1.get_collection("input_y")[0]
		prediction = tf1.get_collection("prediction")[0]
		total_test_pred = 0
		total_correct_pred = 0
		for i in range(int(test_size / batch_size)):
			batch_x, batch_y = feed_data.fetchData(reader, batch_size)
			crp = sess.run(prediction, feed_dict = {input_x: batch_x, input_y : batch_y})
			total_correct_pred += crp
			total_test_pred += batch_size
	print("Accuracy : ", total_correct_pred, "/", total_test_pred)
evaluation()