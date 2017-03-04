# PAMAP

The reference paper is "Time Series Classification Using Multi-Channels Deep CNN"

1. get the database of PAMAP2, which is available at https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring

2. select the data

	Subject: #1~#7 (#8 and #9 either lack the following activities or have diﬀerent dominant hand/foot.)
	
	Activity: 3(standing), 4(walking), 12(ascending stairs), 13(descending stairs)
	
	Data: there are 3 IMUs (hand, chest, ankle) and we cannot determine which one is better for classification, so we should test all three.
	
	We use 3d-acceleration data rather than gyroscope because gyro data is characteristic-less for at least chest position
	
	So the data of one subject has the format of:
	
		Rawdata[1,2,5,6,7,22,23,24,39,40,41]
		
		which is [time, activityID, hand_acc_x,y,z, chest_acc_x,y,z, ankle_acc_x,y,z] 
		
	Then we put the data from 7 subjects into 1 cell.
	
		subject_data={subject1,subject2,subject3,subject4,subject5,subject6,subject7} 
		
3. preprocessing (see the Matlab code attached)

	We normalize each dimension of 3D time series as (x−μ)/σ , where μ and σ are mean and standard deviation of time series. (as in processed3.mat)
	
	Then we apply the sliding window algorithm to extract subsequences from 3D time series with different sliding steps. 
	
		step size: 128, 64, 32, 16, 8 in the reference paper, we only test 128 to reduce calculation time
		
		window length: 256
		
	Now we have:
	
		input_data{1…7}{1…9}, where input_data{i}{j} is a matrix of Nx257, which is N observation*(1d label + 256d data)
		
		 (i for subject 1~7, j for acceleration data 1~9)
		 
	The result is saved as input_data.mat
	
4. divide training and test set (see the Python code attached)

	To evaluate the performance of different models, we adopt the leave-one-out cross validation (LOOCV) technique. Specifically, each time we use one subject’s physical activities as test data, and the physical activities of remaining subjects as training data. Then we repeat this for every subject.
	
5. result

	CNN with input data of IMU (hand)
	
	train acc: 0.78, 0.76, 0.77, 0.79, 0.81, 0.76, 0.79
	
	test acc:  0.53, 0.77, 0.77, 0.79, 0.56, 0.74, 0.76 (avg 0.70)
	
	CNN with input data of IMU (chest)
	
	train acc: 0.87, 0.89, 0.88, 0.89, 0.88, 0.89, 0.87
	
	test acc:  0.86, 0.74, 0.89, 0.86, 0.90, 0.90, 0.88 (avg 0.86)	
	
	CNN with input data of IMU (ankle)
	
	train acc: 0.91, 0.89, 0.87, 0.88, 0.89, 0.90, 0.90
	
	test acc:  0.83, 0.89, 0.81, 0.90, 0.90, 0.85, 0.81 (avg 0.86)
	
	only use the MLP part of the CNN, with input data of IMU (chest)
	
	train acc: 0.76, 0.79, 0.76, 0.76, 0.78, 0.77, 0.80
	
	test acc:  0.77, 0.20, 0.75, 0.75, 0.54, 0.74, 0.39 (avg 0.59)
	
	so the CNN model has a 0.27 accuracy boost compared to simple MLP
