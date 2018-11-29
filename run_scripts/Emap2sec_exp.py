import tensorflow as tf
#sess = tf.InteractiveSession()
import numpy as np
import sys
import random
import sklearn
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
import os
import pandas
import io

flags = tf.app.flags
FLAGS = flags.FLAGS
test_file_count2=1
x_test_arr2 = []
y_test_arr2 = []
test_data2 = []
activArr=[]
activYArr=[]
#flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
#flags.DEFINE_integer('num_epochs', 2, 'Number of epochs to run trainer.')
#flags.DEFINE_integer('batch_size', 100, 'Batch size.')
#flags.DEFINE_integer('min_after_dequeue', 1000,'Min')

def _float_feature(value):
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
	
def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value])) 

def convertOneHot(data):
    print("here")
    y=np.array([int(i[0]) for i in data])
    y_onehot=[0]*len(y)
    for i,j in enumerate(y):
        y_onehot[i]=[0]*(y.max() + 1)
        y_onehot[i][j]=1
    print("exit mod")
    return (y,y_onehot)

def convertOneHot_test(data):
    y=np.array([int(i[0]) for i in data])
    y_onehot=[0]*len(y)
    for i,j in enumerate(y):
        y_onehot[i]=[0]*(y.max() + 1)
        y_onehot[i][j]=1
    return (y,y_onehot)
	
	
def weight_variable(shape):
    W1 = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(W1)

def bias_variable(shape):
	b1 = tf.constant(0.1, shape=shape)
	return tf.Variable(b1)

def conv3d(x, W):
	return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

def max_pool_2x2x2(x):
	return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                        strides=[1, 2, 2, 2, 1], padding='SAME')
def avg_pool_2x2x2(x):
	return tf.nn.avg_pool3d(x, ksize=[1, 2, 2, 2, 1],
                        strides=[1, 2, 2, 2, 1], padding='SAME')

def read_csv(recordsFile):
	writefilename = os.path.join("/net/kihara/smaddhur/tensorFlow/", recordsFile + 'te.tfrecords')
	print('before data read')	
	data = pandas.read_csv(sys.argv[1]).values
	#data = np.genfromtxt(sys.argv[1],delimiter=',')  # Training data
	#train_data_sample = np.array(random.sample(list(data[0:]),int(sys.argv[2])))
	print('Written to csv')	
	train_data_sample = list(data[0:])
	#print(train_data_sample)
	labelarr = np.array([int(i[0]) for i in train_data_sample])
	print('labels done')	
	x_train = np.array([ i[1:len(i)] for i in train_data_sample])
#np.rot90(np.reshape(a,(2,2,2)))
	#x_reshaped = np.zeros((int(sys.argv[2]),11,11,11))
	#x_rot90 = np.zeros((int(sys.argv[2]),11,11,11))
	'''x_reshaped = np.empty((0,11,11,11))
	x_rot90 = np.empty((0,11,11,11))
	x_rot180 = np.empty((0,11,11,11))
	x_rot270 = np.empty((0,11,11,11))	
	count=0
	for i in x_train:
		#print(i)
		x_reshaped = np.append(x_reshaped,[np.reshape(i,(11,11,11))],axis=0)
		x_rot90 = np.append(x_rot90,[np.rot90(x_reshaped[count])],axis=0)
		x_rot180 = np.append(x_rot180,[np.rot90(x_rot90[count])],axis=0)
		x_rot270 = np.append(x_rot270,[np.rot90(x_rot180[count])],axis=0)
		count+=1
	print(np.shape(x_train))
	print(np.shape(x_rot90))
	print(np.shape(x_rot180))
	print(np.shape(x_rot270))
	#rot180 = np.rot90(rot90)
	#rot270 = np.rot90(rot180)
	rot_90 = np.empty((0,int(sys.argv[3])))
	rot_180 = np.empty((0,int(sys.argv[3])))
	rot_270 = np.empty((0,int(sys.argv[3])))
	for i in x_rot90:	
		rot_90 = np.append(rot_90,[np.reshape(i,(int(sys.argv[3])))],axis=0)
	for i in x_rot180:
		rot_180 = np.append(rot_180,[np.reshape(i,(int(sys.argv[3])))],axis=0)
	for i in x_rot270:	
		rot_270 = np.append(rot_270,[np.reshape(i,(int(sys.argv[3])))],axis=0)
	print(np.shape(rot_90))
	print(np.shape(rot_180))
	print(np.shape(rot_270))
	#rot180 = np.reshape(rot180,(-1,int(sys.argv[3])))
	#rot270 = np.reshape(rot270,(-1,int(sys.argv[3])))
	#x_train1 = np.vstack([x_train,rot90,rot180,rot270])
	x_train1 = np.vstack([x_train,rot_90,rot_180,rot_270])
	print(x_train1[10000])
	print(x_train[100])
	print(rot_90[100])
	print(rot_180[100])
	print(rot_270[100])
	#labelarr = np.hstack([labelarr,labelarr,labelarr,labelarr])
	labelarr = np.hstack([labelarr,labelarr,labelarr,labelarr])
	'''
	print("done reading, size: ,lshape: ")
	print(np.shape(x_train))
	print(np.shape(labelarr))
	writer = tf.python_io.TFRecordWriter(writefilename)
	acount=0
	bcount=0
	ccount=0
	counter=0
	for i in range(int(sys.argv[2])):
	#for i in range(4*int(sys.argv[2])):
	#for i in csv:
		feature = x_train[i].tostring()
		#feature = x_train1[i].tostring()
		if(int(labelarr[i])==2):
			acount+=1
		elif(int(labelarr[i])==1):
			bcount+=1
		else:
			ccount+=1
		if counter%1000==0:
			print(str(counter)+"th iteration")
		#mylabel = csv[0]
		#feature = csv[1:len(csv)]
		features = {
			'myfeatures' : _bytes_feature(feature),
			'label' : _int64_feature(int(labelarr[i]))
		}
		example = tf.train.Example(features=tf.train.Features(feature=features))
		writer.write(example.SerializeToString())
		counter+=1
	writer.close()
	print("Done readcsv : alhpa : %d, beta : %d, none : %d\n"%(acount,bcount,ccount))

def read_csv2(x,y,recordsFile):
	writefilename = os.path.join("/net/kihara/smaddhur/tensorFlow/", recordsFile + '2te.tfrecords')
	print('before data read')	
	np_x = np.array(x)
	labelarr = np.array(y)
	mincount = min(np.count_nonzero(labelarr==0),np.count_nonzero(labelarr==1),np.count_nonzero(labelarr==2))
	idx0 = np.where(labelarr==0)
	idx1 = np.where(labelarr==1)
	idx2 = np.where(labelarr==2)
	idx3 = np.where(labelarr==3)
	idx0_new = random.sample(np.ndarray.tolist(idx0[0]),mincount)
	idx1_new = random.sample(np.ndarray.tolist(idx1[0]),mincount)	
	idx2_new = random.sample(np.ndarray.tolist(idx2[0]),mincount)	
	idx3_new = random.sample(np.ndarray.tolist(idx3[0]),mincount)
	print("mincount:%d,size:%d"%(mincount,len(idx3_new)))
#	print(idx3_new[0])
	print(np.shape(np_x))	
	labelarr_new = np.hstack([labelarr[idx0_new],labelarr[idx1_new],labelarr[idx2_new],labelarr[idx3_new]])
	x_train1 = np.array(np.reshape(np_x,(-1,3*3*3*4)))
	x_train = np.vstack([x_train1[idx0_new,:],x_train1[idx1_new,:], x_train1[idx2_new,:],x_train1[idx3_new,:]])
	temp = np.array([labelarr_new])
	totArr = np.hstack([temp.T,x_train])
	np.random.shuffle(totArr)
	labelarr_new = totArr[:,0]
	x_train = totArr[:,1:np.shape(totArr)[1]]
#np.rot90(np.reshape(a,(2,2,2)))
	#x_reshaped = np.zeros((int(sys.argv[2]),11,11,11))
	#x_rot90 = np.zeros((int(sys.argv[2]),11,11,11))
	
	#print(np.shape(x_train))
	#print(len(y))
	writer = tf.python_io.TFRecordWriter(writefilename)
	acount=0
	bcount=0
	ccount=0
	dcount=0
	counter=0
	for i in range(len(labelarr_new)):
	#for i in range(4*int(sys.argv[2])):
	#for i in csv:
		feature = x_train[i].tostring()
		if(int(labelarr_new[i])==2):
			acount+=1
		elif(int(labelarr_new[i])==1):
			bcount+=1
		elif(int(labelarr_new[i])==0):
			ccount+=1
		else:
			dcount+=1
		if counter%100==0:
			print(str(counter)+"th iteration")
			#print(int(labelarr[i]))
		#mylabel = csv[0]
		#feature = csv[1:len(csv)]
		features = {
			'myfeatures' : _bytes_feature(feature),
			'label' : _int64_feature(int(labelarr_new[i]))
		}
		example = tf.train.Example(features=tf.train.Features(feature=features))
		writer.write(example.SerializeToString())
		counter+=1
	writer.close()
	print("Done readcsv : alhpa : %d, beta : %d, loop : %d, bkg : %d\n"%(acount,bcount,ccount,dcount))


def read_and_decode(filename_queue,feature_size):
	reader = tf.TFRecordReader()
	_, serializedStr = reader.read(filename_queue)
	feature_to_type = {
		'myfeatures' : tf.FixedLenFeature([],tf.string),
		'label': tf.FixedLenFeature([], tf.int64)		
	}
	data = tf.parse_single_example(serializedStr, feature_to_type)
	features1 = tf.decode_raw(data['myfeatures'],tf.float64)
	features1.set_shape([feature_size])
	
	#_,label = convertOneHot(data['label'],);
	label = tf.cast(data['label'],tf.int32)
	features1 = tf.cast(features1,tf.float32)

	return features1,label
	
def read_records(filename,batch_size,num_epochs):
	#if not num_epochs: num_epochs = None
	readfile = os.path.join("/net/kihara/smaddhur/tensorFlow/", filename + 'te.tfrecords')
	#with tf.name_scope('input'):
	filename_queue = tf.train.string_input_producer(
	[readfile], num_epochs=num_epochs)

	features2,label = read_and_decode(filename_queue,int(sys.argv[3]))
	#print(features2)
	min_after_dequeue = 100
	capacity = min_after_dequeue + 2 * batch_size
	example_batch, label_batch = tf.train.shuffle_batch(
	[features2, label], batch_size=batch_size, num_threads=2, capacity=capacity,
	min_after_dequeue=min_after_dequeue)
	print("done read rewcords\n")
	return example_batch,label_batch	

def read_records2(filename,batch_size,num_epochs):
	#if not num_epochs: num_epochs = None
	readfile = os.path.join("/net/kihara/smaddhur/tensorFlow/", filename + '2te.tfrecords')
	#with tf.name_scope('input'):
	filename_queue2 = tf.train.string_input_producer(
	[readfile], num_epochs=num_epochs)

	features2,label2 = read_and_decode(filename_queue2,3*3*3*4)
	#print(features2)
	min_after_dequeue = 100
	capacity = min_after_dequeue + 2 * batch_size
	example_batch2, label_batch2 = tf.train.shuffle_batch(
	[features2, label2], batch_size=batch_size, num_threads=2, capacity=capacity,
	min_after_dequeue=min_after_dequeue)
	print("done read rewcords 2\n")
	return example_batch2,label_batch2	

#data = np.genfromtxt(sys.argv[1],delimiter=',')  # Training data
#train_data_sample = np.array(random.sample(list(data),10000))
#x_train=np.array([ i[1:len(i)] for i in train_data_sample])
#y_train,y_train_onehot = convertOneHot(train_data_sample)


def train():
	
	#read_csv(sys.argv[1])

	#no_of_features = final_features.shape[1]-1
	#no_of_labels = 2
	
	
	#y_pred = tf.placeholder(tf.float32, shape=[None,1])
	with tf.Graph().as_default():
		final_features, label = read_records(sys.argv[1],batch_size=100,
								num_epochs=int(sys.argv[4]))
		#x = tf.placeholder(tf.float32, shape=[None, 1331])
		#y_ = tf.placeholder(tf.int64, shape=[1])
		y_test_arr = []
		x_test_arr1 = []
		test_data = []
		test_data_1 =[]
		test_data_11 =[]
		reshaped = []
		reshaped2 = []
		fCoords=[]
		fCoords_tr=[]
		test1 = os.path.join("/net/kihara/smaddhur/tensorFlow/" , sys.argv[5])
		fil1 = open(test1,'r')
		j=0
		filnames=[]
		filnames2=[]
		#myfile=sys.argv[8]
		for filname in fil1:
			print(filname.rstrip())
			filnames.append(filname)

			'''datadf=[]
			indexdf=[]
			countdf=0
			for line in open(filname.rstrip(), 'r'):
				if line.rstrip() == '-1':
					indexdf.append(countdf-1)
				elif not line.rstrip() == '':
					datadf.append(line.rstrip().split(','))
				countdf+=1
			df = pandas.DataFrame(datadf)'''
			df = pandas.DataFrame([line.rstrip().split(',') for line in open(filname.rstrip(), 'r') if not line.rstrip() == ''])
			if(df.empty):
				print("~~~Skipping : %s~~~~~",filname)		
				continue

			pre_td = pandas.read_csv(io.StringIO(u""+df.to_csv(index=False))).values
			print(np.shape(pre_td))			
			
			#pre_td = pandas.read_csv(io.StringIO(u""+df.to_csv(index=False))).values			
			print(np.shape(pre_td))		
			if(np.shape(pre_td)[1] < int(sys.argv[3])):
				print("~~~Skipping : %s~~~~~",filname)		
				continue
			'''if((np.shape(pre_td)[0]-1) != int(pre_td[0][0])*int(pre_td[0][1])*int(pre_td[0][2])):
				print('!!!!!!!!!SHOULD Skip : %s!!!!!!!!!!!',filname)
				continue'''

			test_data.append(pre_td)
			#test_data_11.append(indexdf)

			reshaped.append((int(test_data[j][0][0]),int(test_data[j][0][1]),int(test_data[j][0][2])))
			print(reshaped[j])

			test_tmp = []
#			print(test_data[j][1:len(test_data[j])])
			'''posCount=0
			for i in test_data[j][1:len(test_data[j])]:
				if i[0] != -1:
					test_tmp.append(i)
					posCount+=1
			print("Poscount:%s",(str(posCount)))'''
			coords=[]
			for i in test_data[j][1:len(test_data[j])]:
				if i[3] == -1:
					i[3]=3

				test_tmp.append(i[3:])
				coords.append(i[0:3])				
			fCoords_tr.append(coords)

#			if(posCount!=reshaped[j][0]*reshaped[j][1]*reshaped[j][2]):
#				print('!!!!!!!!!SHOULD Skip : %s!!!!!!!!!!!',filname)
			test_data_sample = np.array(list(test_tmp))
#			test_data_sample = np.array(list(test_data[j]))
			#if(len(i))
			x_test_arr = np.array([ i[1:len(i)] for i in test_data_sample])
			if(np.shape(x_test_arr)[0] == 0):	
				j+=1
				continue
			print(np.shape(x_test_arr)[0])
			x_test_arr1.append(x_test_arr)
			y_testi,_ = convertOneHot_test(test_data_sample)
			y_test_arr.append(y_testi)
			print("Prepared arr:")
#			print(x_test_arr[0])
			j+=1
		
		test_file_count1 = j
		fil1.close()

		test2 = os.path.join("/net/kihara/smaddhur/tensorFlow/" , sys.argv[6])
		fil2 = open(test2,'r')
		j=0
		#wrFile=open(myfile,'w+')
		for filname in fil2:
			filnames2.append(filname)
			'''datadf=[]
			indexdf=[]
			countdf=0
			for line in open(filname.rstrip(), 'r'):
				if line.rstrip() == '':
					continue				
				elif line.rstrip() == '-1':
					indexdf.append(countdf-1)
				elif not line.rstrip() == '':
					datadf.append(line.rstrip().split(','))
				countdf+=1
			df = pandas.DataFrame(datadf)'''

			df = pandas.DataFrame([line.rstrip().split(',') for line in open(filname.rstrip(), 'r') if not line.rstrip() == ''])
			if(df.empty):
				print("~~~Skipping : %s~~~~~",filname)		
				continue
			pre_td = pandas.read_csv(io.StringIO(u""+df.to_csv(index=False))).values
			print(np.shape(pre_td))			

			#pre_td = pandas.read_csv(io.StringIO(u""+df.to_csv(index=False))).values
			#print(np.shape(pre_td))			
			#print(len(indexdf))			
			if(np.shape(pre_td)[1] < int(sys.argv[3])):
				print("~~~Skipping : %s~~~~~",filname)	
				continue
			'''if((np.shape(pre_td)[0]-1) != int(pre_td[0][0])*int(pre_td[0][1])*int(pre_td[0][2])):
				print('!!!!!!!!!SHOULD Skip : %s!!!!!!!!!!!',filname)
				continue'''
			test_data2.append(pre_td)
			#test_data_1.append(indexdf)
			#rArray = str(test_data[j][0]).split(',')
			#reshaped.append((rArray[0],rArray[1],rArray[2]))
			reshaped2.append((int(test_data2[j][0][0]),int(test_data2[j][0][1]),int(test_data2[j][0][2])))
			'''if(np.shape(pre_td)[0]+len(indexdf)-1 != reshaped2[j][0]*reshaped2[j][1]*reshaped2[j][2]):
				print("!!!!!!!!!!!!!!REPORT!!!!!!!!!!!!!! : %s"%(filname))'''
			test_tmp = []
			sum1=0
			sum2=0
			sum3=0
			count1=0
			arr1  =[]
			count2=0
			arr2  =[]
			count3=0
			arr3  =[]

			coords=[]
			for i in test_data2[j][1:len(test_data2[j])]:
				if i[3] == -1:
					i[3]=3
#				else:
#					wrFile.write(",".join(map(str,i[3:]))+"\n")
				if(i[0]==0):	
					count1+=1
					sum1+=np.sum(i)
					arr1.append(np.sum(i[1:]))

				if(i[0]==1):	
					count2+=1
					sum2+=np.sum(i)
					arr2.append(np.sum(i[1:]))

				if(i[0]==2):	
					count3+=1
					sum3+=np.sum(i)
					arr3.append(np.sum(i[1:]))

			#print(np.sum(i))
##			print(count1)
			#print(sum)
				test_tmp.append(i[3:])
				coords.append(i[0:3])				
				#posCount+=1
			#print("Poscount:%s",(str(posCount)))
#			print(coords)
			print(np.mean(arr1))			
			print(np.mean(arr2))	
			print(np.mean(arr3))	
			fCoords.append(coords)
#				if(i[0]==2):	
#					count1+=1
#					sum+=np.sum(i)
#					arr1.append(np.sum(i))
#					print(np.sum(i))
#			print(np.mean(arr1))			
##			print(count1)
			#print(sum)
#			test_data_sample = np.array(test_tmp)
##			test_data_sample = np.array(list(test_data[j]))
			#if(len(i))
			x_test_arr2.append(np.array([ i[1:len(i)] for i in test_tmp]))
			print(np.shape(x_test_arr2[j]))
			if(np.shape(x_test_arr2[j])[0] == 0):	
				j+=1
				continue
			y_testi2,_ = convertOneHot_test(test_tmp)
			y_test_arr2.append(y_testi2)
			j+=1
		#wrFile.close()
		test_file_count2 = j
		fil2.close()
		
		lambdA = tf.placeholder(tf.float32)
		beta = tf.placeholder(tf.float32)
		beta_FS = tf.placeholder(tf.float32)
		W_conv1 = weight_variable([4, 4, 4, 1, 32])
		b_conv1 = bias_variable([32])

		x_image = tf.reshape(final_features, [-1,11,11,11,1])
		#print(np.shape(x_image))
		h_conv1 = tf.nn.relu(conv3d(x_image,W_conv1) + b_conv1)
		#h_conv1 = tf.nn.relu(conv3d(x_image,tf.mul(lambdA,tf.nn.l2_loss(W_conv1))) + b_conv1)
		#h_pool11 = max_pool_2x2x2(h_conv1)

		W_conv2 = weight_variable([3, 3, 3, 32, 64])
		b_conv2 = bias_variable([64])
		#print(W_conv2)
		h_conv2 = tf.nn.relu(conv3d(h_conv1, W_conv2) + b_conv2)
		#h_conv2 = tf.nn.relu(conv3d(h_pool11, W_conv2) + b_conv2)
		#h_pool2 = max_pool_2x2x2(h_conv2)


		W_conv3 = weight_variable([3, 3, 3, 64, 64])
		b_conv3 = bias_variable([64])
		h_conv3 = tf.nn.relu(conv3d(h_conv2, W_conv3) + b_conv3)
		#h_pool3 = max_pool_2x2x2(h_conv3)

		W_conv4 = weight_variable([3, 3, 3, 64, 128])
		b_conv4 = bias_variable([128])
		h_conv4 = tf.nn.relu(conv3d(h_conv3, W_conv4) + b_conv4)

		#W_conv4 = weight_variable([3, 3, 3, 128, 128])
		#b_conv4 = bias_variable([128])
		#h_conv4 = tf.nn.relu(conv3d(h_conv3, W_conv4) + b_conv4)

		W_conv5 = weight_variable([3, 3, 3, 128, 128])
		b_conv5= bias_variable([128])
		h_conv5 = tf.nn.relu(conv3d(h_conv4, W_conv5) + b_conv5)

		W_conv6 = weight_variable([3, 3, 3, 128, 128])
		b_conv6 = bias_variable([128])
		h_conv6 = tf.nn.relu(conv3d(h_conv5, W_conv6) + b_conv6)

		W_conv7 = weight_variable([3, 3, 3, 128, 128])
		b_conv7 = bias_variable([128])
		h_conv7 = tf.nn.relu(conv3d(h_conv6, W_conv7) + b_conv7)

		W_conv8 = weight_variable([3, 3, 3, 128, 128])
		b_conv8 = bias_variable([128])
		h_conv8 = tf.nn.relu(conv3d(h_conv7, W_conv8) + b_conv8)

		W_conv9 = weight_variable([3, 3, 3, 128, 128])
		b_conv9 = bias_variable([128])
		h_conv9 = tf.nn.relu(conv3d(h_conv8, W_conv9) + b_conv9)

		W_conv10 = weight_variable([3, 3, 3, 128, 512])
		b_conv10 = bias_variable([512])
		h_conv10 = tf.nn.relu(conv3d(h_conv9, W_conv10) + b_conv10)

		W_conv11 = weight_variable([3, 3, 3, 512, 512])
		b_conv11 = bias_variable([512])
		h_conv11 = tf.nn.relu(conv3d(h_conv10, W_conv11) + b_conv11)

		W_conv12 = weight_variable([3, 3, 3, 512, 512])
		b_conv12 = bias_variable([512])
		h_conv12 = tf.nn.relu(conv3d(h_conv11, W_conv12) + b_conv12)

		W_conv13 = weight_variable([3, 3, 3, 512, 512])
		b_conv13 = bias_variable([512])
		h_conv13 = tf.nn.relu(conv3d(h_conv12, W_conv13) + b_conv13)

		W_conv14 = weight_variable([3, 3, 3, 512, 512])
		b_conv14 = bias_variable([512])
		h_conv14 = tf.nn.relu(conv3d(h_conv13, W_conv14) + b_conv14)

		W_conv15 = weight_variable([3, 3, 3, 512, 512])
		b_conv15 = bias_variable([512])
		h_conv15 = tf.nn.relu(conv3d(h_conv14, W_conv15) + b_conv15)

		W_conv16 = weight_variable([3, 3, 3, 512, 128])
		b_conv16 = bias_variable([128])
		h_conv16 = tf.nn.relu(conv3d(h_conv15, W_conv16) + b_conv16)


		keep_prob = tf.placeholder(tf.float32)
		#h_pool1 = max_pool_2x2x2(h_conv16)
		h_pool1 = max_pool_2x2x2(h_conv5)
		#W_fc1 = weight_variable([3 * 3 * 3 * 64, 128])
		#W_fc1 = weight_variable([2 * 2 * 2 * 32, 128])
		W_fc1 = weight_variable([6 * 6 * 6 * 128, 1024])
		b_fc1 = bias_variable([1024])
		#h_pool3_flat = tf.reshape(h_pool2, [-1, 3*3*3*64])
		#h_pool3_flat = tf.reshape(h_pool3, [-1, 2*2*2*32])
		h_pool3_flat = tf.reshape(h_pool1, [-1, 6*6*6*128])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
		#W_fc1 = weight_variable([3 * 3 * 3 * 32, 1024])
		#b_fc1 = bias_variable([1024])
		#h_pool1_flat = tf.reshape(h_pool1, [-1, 6*6*6*32])
		#h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

		W_fc2 = weight_variable([1024, 256])
		b_fc2 = bias_variable([256])
		h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
		#h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

		#W_fc3 = weight_variable([256, 128])
		#b_fc3 = bias_variable([128])
		#h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
		#h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

		#W_fc4 = weight_variable([128, 128])
		#b_fc4 = bias_variable([128])
		#h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)
		#h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)
		

		#h_fc2_drop = tf.nn.dropout(h_fc4, keep_prob)

		W_fc5 = weight_variable([256, 3])
		b_fc5 = bias_variable([3])
		#y_conv=tf.matmul(h_fc2_drop, W_fc5) + b_fc5
		y_conv=tf.matmul(h_fc2, W_fc5) + b_fc5
		
		#keep_prob = tf.placeholder(tf.float32)
		#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#		W_fc2 = weight_variable([1024, 3])
#		b_fc2 = bias_variable([3])
#		y_conv=tf.matmul(h_fc1_drop, W_fc2) + b_fc2
		
		label = tf.to_int64(label)
		#_,label_one_hot = convertOneHot(tf.Session().run(label))
		#cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv, label)+lambdA * tf.nn.l2_loss(W_conv1)+lambdA * tf.nn.l2_loss(b_conv1)+lambdA * tf.nn.l2_loss(W_conv2)+lambdA * tf.nn.l2_loss(b_conv2)+lambdA * tf.nn.l2_loss(W_conv3)+lambdA * tf.nn.l2_loss(b_conv3)+lambdA * tf.nn.l2_loss(W_conv4)+lambdA * tf.nn.l2_loss(b_conv4)+lambdA * tf.nn.l2_loss(W_conv5)+lambdA * tf.nn.l2_loss(b_conv5)+lambdA * tf.nn.l2_loss(W_fc1)+lambdA * tf.nn.l2_loss(b_fc1))
		regularizer1 = beta * (tf.abs(W_fc1)+tf.abs(b_fc1))
		#regularizer2 = beta * (tf.abs(W_fc2)+tf.abs(b_fc2))
		#regualaizerl2 = beta * (tf.nn.l2_loss(W_fc1)+tf.nn.l2_loss(b_fc1))
		#regularizer2 = lambdA * (tf.nn.l2_loss(W_conv16)+tf.nn.l2_loss(b_conv16))
		regularizer2 = lambdA * (tf.nn.l2_loss(W_conv5)+tf.nn.l2_loss(b_conv5))
		#regularizer3 = lambdA * (tf.nn.l2_loss(W_conv6)+tf.nn.l2_loss(b_conv6))
		#regularizer4 = lambdA * (tf.nn.l2_loss(W_conv7)+tf.nn.l2_loss(b_conv7))
		#regularizer5 = lambdA * (tf.nn.l2_loss(W_conv8)+tf.nn.l2_loss(b_conv8))
		cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv, label)+regularizer2)+tf.reduce_sum(regularizer1)#+tf.reduce_sum(regualaizerl2))# #+ tf.reduce_sum(regularizer2)
		#cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv, label))#+regularizer5)+tf.reduce_sum(regularizer1)
		train_step =  tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
		correct_prediction = tf.equal(tf.argmax(y_conv,1), label)
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


		x_test = tf.placeholder(tf.float32, shape=[None, int(sys.argv[3])])
		y_test = tf.placeholder(tf.int64, shape=[None, 1])

		x_test = tf.cast(x_test,tf.float32)
		x_image_test = tf.reshape(x_test, [-1,11,11,11,1])
		h_conv1 = tf.nn.relu(conv3d(x_image_test, W_conv1) + b_conv1)
		#h_pool11 = max_pool_2x2x2(h_conv1)
		h_conv2 = tf.nn.relu(conv3d(h_conv1, W_conv2) + b_conv2)
		#h_conv2 = tf.nn.relu(conv3d(h_pool11, W_conv2) + b_conv2)
		#h_pool2 = max_pool_2x2x2(h_conv2)
		#h_pool2_flat = tf.reshape(h_pool2, [-1, 3*3*3*64])
		h_conv3 = tf.nn.relu(conv3d(h_conv2, W_conv3) + b_conv3)
		h_conv4 = tf.nn.relu(conv3d(h_conv3, W_conv4) + b_conv4)
		h_conv5 = tf.nn.relu(conv3d(h_conv4, W_conv5) + b_conv5)
		h_conv6 = tf.nn.relu(conv3d(h_conv5, W_conv6) + b_conv6)
		h_conv7 = tf.nn.relu(conv3d(h_conv6, W_conv7) + b_conv7)
		h_conv8 = tf.nn.relu(conv3d(h_conv7, W_conv8) + b_conv8)
		h_conv9 = tf.nn.relu(conv3d(h_conv8, W_conv9) + b_conv9)
		h_conv10 = tf.nn.relu(conv3d(h_conv9, W_conv10) + b_conv10)
		h_conv11 = tf.nn.relu(conv3d(h_conv10, W_conv11) + b_conv11)
		h_conv12 = tf.nn.relu(conv3d(h_conv11, W_conv12) + b_conv12)
		h_conv13 = tf.nn.relu(conv3d(h_conv12, W_conv13) + b_conv13)
		h_conv14 = tf.nn.relu(conv3d(h_conv13, W_conv14) + b_conv14)
		h_conv15 = tf.nn.relu(conv3d(h_conv14, W_conv15) + b_conv15)
		h_conv16 = tf.nn.relu(conv3d(h_conv15, W_conv16) + b_conv16)
		#h_pool1 = max_pool_2x2x2(h_conv16)
		h_pool1 = max_pool_2x2x2(h_conv5)
		#h_pool3_flat = tf.reshape(h_pool3, [-1, 2*2*2*32])
		h_pool3_flat = tf.reshape(h_pool1, [-1, 6*6*6*128])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
		h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
		#h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
		#h_fc4 = tf.nn.relu(tf.matmul(h_fc3, W_fc4) + b_fc4)
		#h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
		#y_conv_test=tf.matmul(h_fc2, W_fc3) + b_fc3
		y_conv_test=tf.matmul(h_fc2, W_fc5) + b_fc5
		#y_conv_test=tf.matmul(h_fc2, W_fc8) + b_fc8
		#y_test = tf.to_int64(y_test)
		test_correct_prediction = tf.equal(tf.argmax(y_conv_test,1), y_test[0])
		test_accuracy = tf.reduce_mean(tf.cast(test_correct_prediction, tf.float32))
		activStep=h_fc2		
		y_pred = tf.argmax(y_conv_test,1)
		
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		
		try:
			iter=0
			while not coord.should_stop():
				#print("here")
				#print(sess.run(tf.shape(final_features),feed_dict={keep_prob:0.5,lambdA:0.001,beta:10}))
				#print(sess.run(tf.shape(x_image),feed_dict={keep_prob:0.5,lambdA:0.001,beta:10}))
				sess.run(train_step,feed_dict={keep_prob:0.5,lambdA:0.01,beta:100})
				#print(h_conv1.eval(session=sess,feed_dict={keep_prob:0.5,lambdA:0.1}))
				if iter%100 == 0:
					print('Step %d: accuracy = %.2f' % (iter, accuracy.eval(session=sess,feed_dict={keep_prob:0.5,lambdA:0.01,beta:100})))
				iter += 1
		except tf.errors.OutOfRangeError:
			print('Done training for %d steps.' % (iter))
			#coord.request_stop(e)
		finally:
			coord.request_stop()
		coord.join(threads)
		x_train2 = []
		y_train2 = []
		
		for j in range(test_file_count1):
			#test_data_reshape = np.reshape(test_data[j][1:len(test_data[j])],reshaped[j])
			y_test_arr_reshape = np.reshape(y_test_arr[j],(len(y_test_arr[j]),1))	
			print("------SIZE---------")			
			print(np.shape(x_test_arr1[j]))
			#print("test accuracy "+str(j)+":,%g"%test_accuracy.eval(session=sess,feed_dict={x_test:x_test_arr1[j],y_test:y_test_arr_reshape}))	
			#truE-Labels = y_test.eval(session=sess,feed_dict={x_test:x_test_arr[j],y_test:y_test_arr_reshape})
			y_prob = []
			#y_prob.append(sess.run(tf.nn.softmax(y_conv_test),feed_dict={x_test:x_test_arr1[j],y_test:y_test_arr_reshape}))
			#pred_labels = y_pred.eval(session=sess,feed_dict={x_test:x_test_arr1[j],y_test:y_test_arr_reshape})
			fil1 = open('outFileE2_L1'+str(j),'w');
			fil1.write("#"+filnames[j])
			#fil1.write("\n")
			for i in range(len(y_test_arr_reshape)):
				fil1.write(str(y_test_arr_reshape[i][0]));
				fil1.write(';');

			fil1.write('\n')
			indProb=[]
			sInd=0
			
			print(len(x_test_arr1[j]))
			print(len(y_test_arr[j]))
			for ind in range(0,len(y_test_arr[j]),100):
				sInd = min(len(y_test_arr[j]),ind+100)

				temProb = sess.run(tf.nn.softmax(y_conv_test),feed_dict={x_test:x_test_arr1[j][ind:sInd],y_test:y_test_arr_reshape[ind:sInd]})

				if(indProb==[]):
					indProb=temProb
				else:
					indProb = np.ndarray.tolist(np.vstack([np.array(indProb),temProb]))
#				print(indProb)
				pred_labels = y_pred.eval(session=sess,feed_dict={x_test:x_test_arr1[j][ind:sInd],y_test:y_test_arr_reshape[ind:sInd]})
				for i in range(len(pred_labels)):
					fil1.write(str(pred_labels[i]));
					fil1.write(';')

			fil1.close()
			y_prob.append(indProb)
			print(np.shape(y_prob[0]))

#			print("confusion_matrix:"+str(j))
#			print(sklearn.metrics.confusion_matrix(y_test_arr_reshape, pred_labels))

			index=0
			#y_prob_mat = []
			#y_prob_tlabel=[]
			#print(np.shape(test_data[j]))
#			y_prob_mat = np.zeros((len(test_data[j][1:len(test_data[j])])+len(test_data_11[j]),3))
#			y_prob_tlabel = np.zeros((len(test_data[j][1:len(test_data[j])])+len(test_data_11[j]),1))
#			y_prob_tlabel.fill(3)
			y_prob_reshape = np.zeros((reshaped[j][2],reshaped[j][1],reshaped[j][0],3))
			y_prob_tlabel_reshape = np.zeros((reshaped[j][2],reshaped[j][1],reshaped[j][0]))
			y_prob_tlabel_reshape.fill(3)
			idx_coords=fCoords_tr[j]
#			print(len(np.ndarray.tolist(np.delete(np.array(range(len(test_data[j][1:len(test_data[j])])+len(test_data_11[j]))), np.array(test_data_11[j])))))
#			for idxit in np.ndarray.tolist(np.delete(np.array(range(len(test_data[j][1:len(test_data[j])])+len(test_data_11[j]))), np.array(test_data_11[j]))):
#				y_prob_mat[idxit]=y_prob[0][index]
#				print(y_prob_mat[idxit])
#				y_prob_tlabel[idxit]=y_test_arr_reshape[index]
#				index+=1
			for k in idx_coords:
				'''if(int(k[0])==-1):		#check string
					y_prob_mat.append([0,0,0])
#					y_prob[0] = np.vstack([y_prob[0][:index],np.array([0,0,0]),y_prob[0][index:]]) if y_prob[0][:index].size else np.vstack([np.array([0,0,0]),y_prob[0][index:]])
					y_prob_tlabel.append(3)
				else:'''
				#print(k)
				y_prob_reshape[int(k[0]),int(k[1]),int(k[2])]=y_prob[0][index]
				y_prob_tlabel_reshape[int(k[0]),int(k[1]),int(k[2])]=y_test_arr_reshape[index]
				index+=1



			'''for k in test_data[j][1:len(test_data[j])]:
				if(int(k[0])==-1):		#check string
					y_prob_mat.append([0,0,0])
#					y_prob[0] = np.vstack([y_prob[0][:index],np.array([0,0,0]),y_prob[0][index:]]) if y_prob[0][:index].size else np.vstack([np.array([0,0,0]),y_prob[0][index:]])
					y_prob_tlabel.append(3)
				else:
					y_prob_mat.append(y_prob[0][index])
					y_prob_tlabel.append(y_test_arr_reshape[index])					
					index+=1'''
			#print(y_prob_tlabel)			
			#print((y_prob[0]))
			#bkg_count = np.shape(y_prob[0])[0] - (reshaped[j][0]+reshaped[j][1]+reshaped[j][2])
			#y_prob_reshape = np.reshape(y_prob_mat,(reshaped[j][2],reshaped[j][1],reshaped[j][0],3))
			#y_prob_tlabel_reshape = np.reshape(y_prob_tlabel,(reshaped[j][2],reshaped[j][1],reshaped[j][0]))
			#print(y_prob_reshape[5])
			for a in range(reshaped[j][2]):
				for b in range(reshaped[j][1]):
					for c in range(reshaped[j][0]):				
						line=[]
						bkg_count=0
						for aa in range(-1,2,1):
							for bb in range(-1,2,1):
								for cc in range(-1,2,1):
									if(a+aa < 0 or b+bb < 0 or c+cc < 0 or a+aa >= reshaped[j][2] or b+bb >= reshaped[j][1] or c+cc >= reshaped[j][0]):
										line.append([0,0,0,1])
										bkg_count+=1
									elif(np.array_equal(y_prob_reshape[a+aa][b+bb][c+cc],np.array([0,0,0]))):	
#										if(np.array_equal(y_prob_reshape[a+aa][b+bb][c+cc],np.array([0,0,0])) and y_prob_tlabel_reshape[a+aa][b+bb][c+cc]!=3):
#											print("~~~~~~~~~FLAGGGGGGGGGGG~~~~~~~~~~~~~")								
										line.append([0,0,0,1])
										bkg_count+=1
									else:
										#print(y_prob_reshape[a+aa][b+bb][c+cc])										
										modi_y_prob = y_prob_reshape[a+aa][b+bb][c+cc]
										#print(np.hstack([modi_y_prob,[0]]))
										line.append(np.ndarray.tolist(np.hstack([modi_y_prob,[0]])))
						#line = np.reshape(line,(3,3,3))
						#print(np.array_equal(np.array(line),np.array([0,0,0,0])))
						if bkg_count == 27:
							#print("-------Y_probs------------")
							#print(y_prob_tlabel_reshape[a][b][c])							
							continue
						x_train2.append(line)
						y_train2.append(y_prob_tlabel_reshape[a][b][c])
			print(np.shape(np.array(x_train2)))
			print("-----------Y-------------")
			#print(y_train2)
		read_csv2(x_train2,y_train2,sys.argv[1])
		final_features2, label2 = read_records2(sys.argv[1],batch_size=100,
								num_epochs=int(sys.argv[7]))
		x_test_arr2_full = []
		y_test_arr2_full = []
		for j in range(test_file_count2):
			#test_data_reshape = np.reshape(test_data[j][1:len(test_data[j])],reshaped[j])
			y_test_arr_reshape = np.reshape(y_test_arr2[j],(len(y_test_arr2[j]),1))	
			y_prob = []
#			print(np.shape(x_test_arr2[j]))
			fil1 = open('outFileE2_L2'+str(j),'w');
			filA = open('outFileAct_L2'+str(j),'w');
			fil1.write("#"+filnames2[j])
			filA.write("#"+filnames2[j])
			#fil1.write("\n")

			for i in range(len(y_test_arr_reshape)):
				fil1.write(str(y_test_arr_reshape[i][0]))
				fil1.write(';')
			fil1.write('\n')
			indProb=[]
			sInd=0
			pred_labels=[]
			cum_pred_labels=[]
			#print(len(x_test_arr2[j]))
			#print(len(y_test_arr2[j]))
			for ind in range(0,len(y_test_arr2[j]),100):
				sInd = min(len(y_test_arr2[j]),ind+100)
				#print("test accuracy "+str(j)+":,%g"%test_accuracy.eval(session=sess,feed_dict={x_test:x_test_arr2[j],y_test:y_test_arr_reshape}))	
			#truE_Labels = y_test.eval(session=sess,feed_dict={x_test:x_test_arr[j],y_test:y_test_arr_reshape})
				temProb = sess.run(tf.nn.softmax(y_conv_test),feed_dict={x_test:x_test_arr2[j][ind:sInd],y_test:y_test_arr_reshape[ind:sInd]})
				#print(np.shape(indProb))
				#print(np.shape(temProb))
				activArr = sess.run(activStep,feed_dict={x_test:x_test_arr2[j][ind:sInd],y_test:y_test_arr_reshape[ind:sInd]})
				for aA in activArr:
					#print(",".join(map(str,np.ndarray.tolist(aA)))+"\n")
					filA.write(",".join(map(str,np.ndarray.tolist(aA)))+"\n")
				
				if(indProb==[]):
					indProb=temProb
				else:
					indProb = np.ndarray.tolist(np.vstack([np.array(indProb),temProb]))
#				print(indProb)
				pred_labels = y_pred.eval(session=sess,feed_dict={x_test:x_test_arr2[j][ind:sInd],y_test:y_test_arr_reshape[ind:sInd]})
				for i in range(len(pred_labels)):
					cum_pred_labels.append(pred_labels[i])
					fil1.write(str(pred_labels[i]));
					fil1.write(';')
			print("---------CUM {RED")
			print(len(cum_pred_labels))
			print("-----idx_coords-----")
			print(len(idx_coords))
			fil1.close()
			filA.close()			
			#print(indProb[0])
			y_prob.append(indProb)
			print(np.shape(y_prob[0]))
#			print("confusion_matrix:"+str(j))
#			print(sklearn.metrics.confusion_matrix(y_test_arr_reshape, pred_labels))

			index=0
			#y_prob_mat = []
			#y_prob_tlabel2=[]
			'''y_prob_mat = np.zeros((reshaped2[j][2]*reshaped2[j][1]*reshaped2[j][0],3))
			y_prob_tlabel2 = np.zeros((reshaped2[j][2]*reshaped2[j][1]*reshaped2[j][0],1))
			y_prob_tlabel2.fill(3)'''
			y_prob_reshape = np.zeros((reshaped2[j][2],reshaped2[j][1],reshaped2[j][0],3))
			y_prob_tlabel_reshape2 = np.zeros((reshaped2[j][2],reshaped2[j][1],reshaped2[j][0]))
			y_prob_tlabel_reshape2.fill(3)
			y_prob_truelabel_reshape2 = np.zeros((reshaped2[j][2],reshaped2[j][1],reshaped2[j][0]))
			y_prob_tlabel_reshape2.fill(3)
			idx_coords=fCoords[j]
			#print(np.delete(np.array(range(len(test_data2[j][1:len(test_data2[j])])+len(test_data_1[j]))), np.array(test_data_1[j])))
			#resArr = [res for res in range(len(test_data2[j][1:len(test_data2[j])])+len(test_data_1[j])) if res not in test_data_1[j]]
			#print(len(resArr))
			#print(len(np.ndarray.tolist(np.delete(np.array(range(len(test_data2[j][1:len(test_data2[j])])+len(test_data_1[j]))), np.array(test_data_1[j])))))
			'''for idxit in np.ndarray.tolist(np.delete(np.array(range(len(test_data2[j][1:len(test_data2[j])])+len(test_data_1[j]))), np.array(test_data_1[j]))):
				y_prob_mat[idxit]=y_prob[0][index]
#				print(y_prob_mat[idxit])
				y_prob_tlabel2[idxit]=y_test_arr_reshape[index]
				index+=1'''
			'''for k in range(len(test_data2[j][1:len(test_data2[j])])+len(test_data_1[j])):
				if(k in test_data_1[j]):		#check string
					#y_prob_mat.append([0,0,0])
					y_prob[0] = np.vstack([y_prob[0][:index],np.array([0,0,0]),y_prob[0][index:]]) if y_prob[0][:index].size else np.vstack([np.array([0,0,0]),y_prob[0][index:]])
					y_prob_tlabel2.append(3)
					print(k)
				else:
					print(index)
					y_prob_mat.append(y_prob[0][index])
					y_prob_tlabel2.append(y_test_arr_reshape[index])					
					index+=1'''
			for k in idx_coords:
				'''if(int(k[0])==-1):		#check string
					y_prob_mat.append([0,0,0])
#					y_prob[0] = np.vstack([y_prob[0][:index],np.array([0,0,0]),y_prob[0][index:]]) if y_prob[0][:index].size else np.vstack([np.array([0,0,0]),y_prob[0][index:]])
					y_prob_tlabel.append(3)
				else:'''
				#print(k)
				y_prob_reshape[int(k[0]),int(k[1]),int(k[2])]=y_prob[0][index]
				if(y_test_arr_reshape[index]!=3):
					y_prob_tlabel_reshape2[int(k[0]),int(k[1]),int(k[2])]=cum_pred_labels[index]
				else:
					y_prob_tlabel_reshape2[int(k[0]),int(k[1]),int(k[2])]=y_test_arr_reshape[index]
				y_prob_truelabel_reshape2[int(k[0]),int(k[1]),int(k[2])]=y_test_arr_reshape[index]
				index+=1
			print(np.shape(y_prob_reshape))
			print(np.shape(y_prob_tlabel_reshape2))
			x_test2 = []
			y_test_2 = []
			y_test_true = []
			#print((y_prob[0]))
			#bkg_count = np.shape(y_prob[0])[0] - (reshaped[j][0]+reshaped[j][1]+reshaped[j][2])
			#y_prob_reshape = np.reshape(np.array(y_prob_mat),(reshaped2[j][2],reshaped2[j][1],reshaped2[j][0],3))
			#y_prob_tlabel_reshape2 = np.reshape(y_prob_tlabel2,(reshaped2[j][2],reshaped2[j][1],reshaped2[j][0]))			
#			y_prob_reshape = np.reshape(y_prob_mat,(reshaped2[j][2],reshaped2[j][1],reshaped2[j][0],3))
#			y_prob_tlabel_reshape2 = np.reshape(y_prob_tlabel2,(reshaped2[j][2],reshaped2[j][1],reshaped2[j][0]))			
			print("Reshape done")
			for a in range(reshaped2[j][2]):
				for b in range(reshaped2[j][1]):
					for c in range(reshaped2[j][0]):				
						line=[]
						bkg_count=0
						for aa in range(-1,2,1):
							for bb in range(-1,2,1):
								for cc in range(-1,2,1):
									if(a+aa < 0 or b+bb < 0 or c+cc < 0 or a+aa >= reshaped2[j][2] or b+bb >= reshaped2[j][1] or c+cc >= reshaped2[j][0]):
										line.append([0,0,0,1])
										bkg_count+=1
									elif(np.array_equal(y_prob_reshape[a+aa][b+bb][c+cc],np.array([0,0,0]))):									
										line.append([0,0,0,1])
										bkg_count+=1
									else:
										#print(y_prob_reshape[a+aa][b+bb][c+cc])
										modi_y_prob = y_prob_reshape[a+aa][b+bb][c+cc]
										line.append(np.ndarray.tolist(np.hstack([modi_y_prob,[0]])))
						#line = np.reshape(line,(3,3,3))
						#print(np.array_equal(np.array(line),np.array([0,0,0,0])))
						if bkg_count == 27:
							continue			
						x_test2.append(line)
						#if(y_prob_tlabel_reshape2[a][b][c]!=3):
						#	y_test_2.append(y_prob_tlabel_reshape2[a][b][c][0])
						#else:
						y_test_2.append(y_prob_tlabel_reshape2[a][b][c])							
						y_test_true.append(y_prob_truelabel_reshape2[a][b][c])							
			print(np.shape(np.array(x_test2)))
			print("-----------YT-------------")
			#print(y_test_2)
			x_test_arr2_full.append(x_test2)
			#y_test_arr2_full.append(y_test_2)
			y_test_arr2_full.append(y_test_true)
		sess.close()
			#train2

		
		x_image2 = final_features2
		W_fc21 = weight_variable([3 * 3 * 3 * 4, 1024])
		b_fc21 = bias_variable([1024])
		h_fc21 = tf.nn.relu(tf.matmul(x_image2, W_fc21) + b_fc21)

		W_fc22 = weight_variable([1024, 1024])
		b_fc22 = bias_variable([1024])
		h_fc22 = tf.nn.relu(tf.matmul(h_fc21, W_fc22) + b_fc22)

		W_fc23 = weight_variable([1024, 1024])
		b_fc23 = bias_variable([1024])
		h_fc23 = tf.nn.relu(tf.matmul(h_fc22, W_fc23) + b_fc23)

		W_fc24 = weight_variable([1024, 1024])
		b_fc24 = bias_variable([1024])
		h_fc24 = tf.nn.relu(tf.matmul(h_fc23, W_fc24) + b_fc24)

		W_fc25 = weight_variable([1024, 256])
		b_fc25 = bias_variable([256])
		h_fc25 = tf.nn.relu(tf.matmul(h_fc24, W_fc25) + b_fc25)


		W_fc26 = weight_variable([256, 4])
		b_fc26 = bias_variable([4])
		#y_conv=tf.matmul(h_fc2_drop, W_fc5) + b_fc5
		y_conv2=tf.matmul(h_fc25, W_fc26) + b_fc26

		label2 = tf.to_int64(label2)
		regularizer_FS1 = beta_FS * (tf.abs(W_fc21)+tf.abs(b_fc21)) 
		regularizer_FS2 = beta_FS * (tf.abs(W_fc22)+tf.abs(b_fc22))
		regularizer_FS3 = beta_FS * (tf.abs(W_fc23)+tf.abs(b_fc23))		 
		regularizer_FS4 = beta_FS * (tf.abs(W_fc24)+tf.abs(b_fc24))	 
		cross_entropy2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_conv2, label2)) + tf.reduce_sum(regularizer_FS1) #+ tf.reduce_sum(regularizer_FS2) + tf.reduce_sum(regularizer_FS3) + tf.reduce_sum(regularizer_FS4)
		train_step2 =  tf.train.AdamOptimizer(0.001).minimize(cross_entropy2)
		correct_prediction2 = tf.equal(tf.argmax(y_conv2,1), label2)
		accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))


		x_image_test2 = tf.placeholder(tf.float32, shape=[None, 108])
		y_test2 = tf.placeholder(tf.int64, shape=[None, 1])

		x_image_test2 = tf.cast(x_image_test2,tf.float32)
		x_image_test2 = tf.reshape(x_image_test2, [-1,3*3*3*4])
		h_fc21 = tf.nn.relu(tf.matmul(x_image_test2, W_fc21) + b_fc21)
		#h_pool11 = max_pool_2x2x2(h_fc1)
		h_fc22 = tf.nn.relu(tf.matmul(h_fc21, W_fc22) + b_fc22)
		h_fc23 = tf.nn.relu(tf.matmul(h_fc22, W_fc23) + b_fc23)
		h_fc24 = tf.nn.relu(tf.matmul(h_fc23, W_fc24) + b_fc24)
		h_fc25 = tf.nn.relu(tf.matmul(h_fc24, W_fc25) + b_fc25)

		y_conv_test2=tf.matmul(h_fc25, W_fc26) + b_fc26
		#y_test = tf.to_int64(y_test)
		test_correct_prediction2 = tf.equal(tf.argmax(y_conv_test2,1), y_test2[0])
		test_accuracy2 = tf.reduce_mean(tf.cast(test_correct_prediction2, tf.float32))
		
		y_pred2 = tf.argmax(y_conv_test2,1)
		
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		
		try:
			iter=0
			while not coord.should_stop():
				#print(sess.run(tf.shape(final_features2)))
				sess.run(train_step2,feed_dict={beta_FS:0.1})
				#print(h_conv1.eval(session=sess,feed_dict={keep_prob:0.5,lambdA:0.1}))
				if iter%100 == 0:
					print('Step %d: accuracy = %.2f' % (iter, accuracy2.eval(session=sess,feed_dict={beta_FS:0.1})))
				iter += 1
		except tf.errors.OutOfRangeError:
			print('Done training for %d steps.' % (iter))
			#coord.request_stop(e)
		finally:
			coord.request_stop()
		coord.join(threads)

		print("----FC------")
		print(test_file_count2)
		for j in range(test_file_count2):
			#test_data_reshape = np.reshape(test_data[j][1:len(test_data[j])],reshaped[j])
#			print(len(y_test_arr2_full[j]))
			y_test_arr_reshape = np.reshape(y_test_arr2_full[j],(len(y_test_arr2_full[j]),1))	
			#y_test_arr_reshape = np.reshape(y_test_arr2[j],(len(y_test_arr2[j]),1))	
			#print(y_test_arr_reshape)
			print(np.shape(x_test_arr2_full[j])[0])			
			print(np.shape(y_test_arr_reshape))
			print("test accuracy2 "+str(j)+":,%g"%test_accuracy2.eval(session=sess,feed_dict={x_image_test2:np.reshape(x_test_arr2_full[j],(np.shape(x_test_arr2_full[j])[0],3*3*3*4)),y_test2:y_test_arr_reshape,beta_FS:0.1}))	
			#truE_Labels = y_test.eval(session=sess,feed_dict={x_test:x_test_arr[j],y_test:y_test_arr_reshape})
			y_prob2 = []
			y_prob2.append(sess.run(tf.nn.softmax(y_conv_test2),feed_dict={x_image_test2:np.reshape(x_test_arr2_full[j],(np.shape(x_test_arr2_full[j])[0],3*3*3*4)),y_test2:y_test_arr_reshape,beta_FS:0.1}))
			#print(y_prob2)
			pred_labels2 = y_pred2.eval(session=sess,feed_dict={x_image_test2:np.reshape(x_test_arr2_full[j],(np.shape(x_test_arr2_full[j])[0],3*3*3*4)),y_test2:y_test_arr_reshape,beta_FS:0.1})
			fil1 = open('outFileE2_L2_new'+str(j),'w');
			fil1.write("#"+filnames2[j])
			for i in range(len(y_test_arr_reshape)):
				fil1.write(str(y_test_arr_reshape[i][0]));
				fil1.write(';');

			fil1.write('\n')
			for i in range(len(pred_labels2)):
				fil1.write(str(pred_labels2[i]));
				fil1.write(';');
			fil1.close()
			print("confusion_matrix:"+str(j))
			print(sklearn.metrics.confusion_matrix(y_test_arr_reshape, pred_labels2))

	#	print("Precision %g" % sklearn.metrics.precision_score(y_test.eval(session=sess), y_pred.eval(session=sess)))
	#	print("Recall %g" % sklearn.metrics.recall_score(y_test.eval(session=sess), y_pred.eval(session=sess)))
	#	print("f1_score %g" % sklearn.metrics.f1_score(y_test.eval(session=sess), y_pred.eval(session=sess)))
		
		
		#y_true = np.array(tf.argmax(y_test_onehot,1))
		#print(metrics.confusion_matrix(tf.argmax(y_test_onehot,1),y_pred.eval(session=sess,feed_dict={x: x_test, y_: y_test_onehot, keep_prob: 1.0})))
		#confusion_matrix(sess,correct_prediction,x_test,y_test_onehot,x,y_)
		sess.close()
	
def main(_):
	train()
	
if __name__ == '__main__':
	tf.app.run()



