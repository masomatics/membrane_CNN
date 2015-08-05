import sys,pickle
import numpy as np
from chainer import cuda, Function, FunctionSet, gradient_check, Variable, optimizers
import chainer.functions as F
from matplotlib import pyplot as plt
from PIL import Image
import os
import cPickle as pickle
import scipy.ndimage as ndi

'''
__init__(self, crop = 33,  resol = 256)
train(self, epoch) 
augment_data_batch(self, x_train_batch, y_train_batch)
prepare_dataset(self, datapath_test,  datapath_train ,label_datapath_test, \
                        label_datapath_train, cropsize , boringsize)
'''


class train_cnn_cdd(object):    
    
    def __init__(self, crop = 33,  resol = 256):
        self.crop = crop  
        self.resol = resol
        pass
    
    def train(self, epoch):        
        
        '''Train the convolutional neural network that approximates p(y_i | y_{-i}, x) 
        and save the model as .pkl file in the models/ folder.
        
        Keyword arguments:
        epoch -- number of epoch size        
        '''
        
        
        #################
        ####Prepare Data#### 
        #################
        print '=================================================================='
        print '====This file will use the model file convnet_conditional.py======'
        print '=================================================================='

        cropsize = self.crop
        epoch_size = epoch 
        resolution =  self.resol
        boringsize = 0
        cooling_rate = 0.95
        initial_temperature = 0
        
        print 'Cropsize :%d,  Resolution :%d,  Epoch : %d, Boringsize : %d' %(cropsize, resolution, epoch_size, boringsize)
        print  'cooling_rate:%s, initial temperature:%s, final temperature:%s' %\
        (str(cooling_rate), str(initial_temperature), str(initial_temperature*(cooling_rate**epoch_size)) )

        import sys  
        parentpath= '/home/koyama-m/Research/membrane_CNN/'
        models_path= parentpath + 'models/'
        datapath_train = parentpath +  'data/training_dataset/256_training_dataset_crop%s/' %str(cropsize)
        datapath_test = parentpath + 'data/test_dataset/256_test_dataset_crop%s/' %str(cropsize)
        label_datapath_train = parentpath + 'data/training_dataset/label_256_training_dataset_crop%s/'%str(cropsize)
        label_datapath_test = parentpath +  'data/test_dataset/label_256_test_dataset_crop%s/'%str(cropsize)

        trainingfilepath_prefix = '256_training_image_'   #256_test_image_%3d%3d%3d.tif %(slice_index, row, column)
        label_filepath_prefix = 'label_256_training_image_' 
        sys.path.append(parentpath)
        sys.path.append(models_path)
        sys.path.append(datapath_train)
        sys.path.append(datapath_test)
        sys.path.append(label_datapath_train)
        sys.path.append(label_datapath_test)
        sys.path.append(trainingfilepath_prefix)
        sys.path.append(label_filepath_prefix)                
        
  
        #######################
        ###### Setup############
        #######################
        print 'importing the required modules ... '
        import logistic_reg
        import convnet_conditional as convnet
        reload(logistic_reg)
        reload(convnet)


        print 'COMPLETE \n' 
        ###### load dataset ######
        print 'Loading the preprocessed test data and training data ... '

        pre_x_train, pre_x_train_label, pre_x_test, pre_x_test_label, pre_x_valid,  \
                                pre_x_valid_label,  pre_y_train, pre_y_test, pre_y_valid=\
        self.prepare_dataset(datapath_test,  datapath_train ,label_datapath_test,  label_datapath_train, cropsize , boringsize)
        
        x_train, x_test, x_valid, y_train, y_test, y_valid = \
        self.reshape_for_chainer_cnn(pre_x_train, pre_x_train_label, pre_x_test, pre_x_test_label, pre_x_valid,  \
                                pre_x_valid_label,  pre_y_train, pre_y_test, pre_y_valid) 
                       
        ######### init GPU status #######
        cuda.init()

        #FXN MUST BE DEFINED BEFORE INITIALIZATION 
        ######## init models ########
        model_cpu_ver = convnet.convnet_cdd(patchsize=x_train.shape[2])
        model =  convnet.convnet_cdd(patchsize=x_train.shape[2]).to_gpu()

        ######## init optimizer #######
        print 'Initializing the optimizer...\n '

        optimizer = optimizers.Adam()
        optimizer.setup(model.collect_parameters())
        optimizer.zero_grads()

        print 'Initiating the Training Sequence...'

        #######################
        ######Training###########
        #######################

        import time

        trainsize = x_train.shape[0]
        validsize = x_valid.shape[0]
        start_time = time.time()

        minibatchsize = 50
        temperature = initial_temperature
        
        
        for epoch in xrange(epoch_size):

            print 'Current temperature :'  + str(temperature) 
            start_time = time.time()

            indexes = np.random.permutation(trainsize)
            n_batch = indexes.shape[0]/minibatchsize
            sum_loss = 0
            sum_accuracy = 0
            for i in xrange(0, trainsize, minibatchsize):

                batchrange = indexes[i : i + minibatchsize]
                x_train_batch = x_train[batchrange]
                x_train_batch[:,1,:,:] = self.filter_batch(x_train_batch[:,1,:,:], temperature)
                y_train_batch = y_train[batchrange]
                
                pre_x_batch, pre_y_batch = self.augment_data_batch(x_train_batch, y_train_batch)
                
                x_batch = cuda.to_gpu(pre_x_batch)
                y_batch = cuda.to_gpu(pre_y_batch)                
                
                optimizer.zero_grads()

                loss, accuracy,pred = model.forward(x_batch, y_batch)

                sum_loss += loss.data*minibatchsize
                sum_accuracy += accuracy.data*minibatchsize
                loss.backward()
                optimizer.update()

            sum_val_loss = 0
            sum_val_accuracy = 0
            for i in xrange(0,validsize,minibatchsize):
                                
                x_valid_batch =x_valid[i : i + minibatchsize]
                #x_valid_batch[:,1,:,:] = self.filter_batch(x_valid_batch[:,1,:,:], temperature)
                y_valid_batch =y_valid[i : i + minibatchsize]
                pre_x_batch, pre_y_batch = self.augment_data_batch(x_valid_batch, y_valid_batch)
                            
                x_batch = cuda.to_gpu(pre_x_batch)
                y_batch = cuda.to_gpu(pre_y_batch)
                                
                loss, accuracy,pred = model.forward(x_batch, y_batch,False)
                sum_val_loss += loss.data*minibatchsize
                sum_val_accuracy += accuracy.data*minibatchsize

            elapsed_time = time.time() - start_time
            print 'Cropsize :%d,   Boringsize : %d' %(cropsize,  boringsize)
            print 'Elapsed time is ' + str(elapsed_time)     
            print 'epoch ', epoch
            print 'train loss:' + str(sum_loss/trainsize)
            print 'train accuracy(%)' + str(sum_accuracy/trainsize*100)
            print 'validation loss' + str(sum_val_loss/validsize)    
            print 'validation accuracy(%)' + str(sum_val_accuracy/validsize*100)    
            temperature = temperature * cooling_rate

            #Save the model every epoch...  for the ease of my research
            print type(model)
            modelname = 'hole%s_Anneal%s_conditional_distr_trained_model%s_crop%sepoch%s.pkl' %(str(boringsize), str(cooling_rate),  str(resolution),  str(cropsize), str(epoch_size))
            print modelname
            pickle.dump(model, open(models_path+ modelname,'wb'),-1)

        elapsed_time = time.time() - start_time
        print elapsed_time
        print 'Training sequence COMPLETE'  


        print 'Initiating the Testing Sequence...'

        #######################
        ###### Testing ###########
        #######################
        testsize = x_test.shape[0]
        minibatchsize = 1000
        sum_loss = 0
        sum_accuracy = 0
        confusion_matrix = np.zeros((2,2))
        for i in xrange(0, testsize, minibatchsize):
            

                x_test_batch = x_test[i : i + minibatchsize]
                y_test_batch = y_test[i : i + minibatchsize]
                pre_x_batch, pre_y_batch = self.augment_data_batch(x_test_batch, y_test_batch)
                

                
                x_batch = cuda.to_gpu(pre_x_batch)
                y_batch = cuda.to_gpu(pre_y_batch)
                loss, accuracy, prob = model.forward(x_batch, y_batch, train=False)
                sum_loss += loss.data*minibatchsize
                sum_accuracy += accuracy.data*x_batch.shape[0]
                #pred = cuda.to_cpu(prob.data)[:,0]>threshold
                pred = np.argmax(cuda.to_cpu(prob.data),axis=1)

                #calc confusion matrix
                for j in xrange(x_batch.shape[0]):
                    confusion_matrix[cuda.to_cpu(y_batch)[j],pred[j]] += 1

        print 'Testing sequence COMPLETE... saving the log... '  
        txtname = 'hole%sAnneal%sconditional_distr_trained_model%s_crop%sepoch%s_log.txt' %(str(boringsize), str(cooling_rate), str(resolution),  str(cropsize), str(epoch_size))
        sys.stdout = open(models_path+ txtname,"w")                        
        print 'test loss:' + str(sum_loss/testsize)

        print 'chance lebel(accuracy)' + str((np.sum(confusion_matrix[0,:])/np.sum(confusion_matrix)))
        print 'test accuracy(%)' + str((confusion_matrix[0,0]+confusion_matrix[1,1])/np.sum(confusion_matrix))

        print 'confusion_matrix:'
        print confusion_matrix

        sys.stdout.close()

    def augment_data_batch(self, x_train_batch, y_train_batch):
        '''Augment the data with 4th Dyhedral group action.
        
        Keyword arguments:
        x_train_batch  -- the batch to be augmented with the group action.
        y_train_batch --  label information of the original batch
        '''
        

        x_batch_orig = x_train_batch
        y_batch_orig = y_train_batch
        x_batch_tr = np.transpose(x_train_batch, (0,1,3,2))
        y_batch_tr = y_batch_orig
        x_batch_lr  = x_train_batch[:,:,:,::-1]
        y_batch_lr  = y_batch_orig
        x_batch_ud  = x_train_batch[:,:,::-1,:]
        y_batch_ud  = y_batch_orig
        x_batch_udtr = np.transpose(x_batch_ud, (0,1,3,2)) 
        y_batch_udtr = y_batch_orig

        pre_x_batch = np.float32(np.concatenate((x_batch_orig,x_batch_tr, x_batch_lr, x_batch_ud, x_batch_udtr), axis=0))
        pre_y_batch = np.int32(np.concatenate((y_batch_orig,y_batch_tr, y_batch_lr, y_batch_ud,y_batch_udtr), axis=0))        
        
        return pre_x_batch, pre_y_batch
        
    def prepare_dataset(self, datapath_test,  datapath_train ,label_datapath_test, \
                        label_datapath_train, cropsize , boringsize):
 
        '''Augment the data with 4th Dyhedral group action.
        
        Keyword arguments:
        datapath_test  -- folder that contains all the data.
        datapath_train -- folder that contains training data
        label_datapath_train -- folder
        '''
                        
        import load_preprocessed_data_set as loadt         
        
        dataslice_test = loadt.make_data_set(datapath_test,  patchsize =cropsize ,label_data_path = label_datapath_test, is_training = False, boring_size = boringsize)
        dataslice_train = loadt.make_data_set(datapath_train, patchsize =cropsize ,label_data_path = label_datapath_train, is_training = True, boring_size = boringsize)
        
        pickle.dump({'test': dataslice_test , 'train':dataslice_train}, open('/home/koyama-m/Research/membrane_CNN/data/temp_data/temp_dataset_with_hole.pkl','wb') ) 
        dataset = pickle.load(open('/home/koyama-m/Research/membrane_CNN/data/temp_data/temp_dataset_with_hole.pkl','rb')) 
        
        dataslice_test = dataset['test'] 
        dataslice_train= dataset['train'] 
        
        x_train0 = dataslice_train['x_data']/255.
        x_train0_label = dataslice_train['label_x_data']/255.
        y_train0 = dataslice_train['y_data']
        pre_x_test = dataslice_test['x_data']/255.
        pre_x_test_label = dataslice_test['label_x_data']/255.
        pre_y_test = dataslice_test['y_data']        

        print 'COMPLETE \n '

        ##### Validation Set and  the rest #### 

        print 'Preparing the Validation data... '
        neg_index = np.where(y_train0 == 0)[0]
        pos_index = np.where(y_train0 == 1)[0]
        neg_pos_prop = [neg_index.shape[0], pos_index.shape[0] ]
        
        print 'Pos Neg proportion in training data is' + str(neg_pos_prop)
        
        validate_index = np.arange(150000, y_train0.shape[0],1)
        train_index  = np.arange(0,150000,1)
        print 'Validation data size is  ' + str(validate_index.shape[0])

        pre_x_valid = x_train0[validate_index]
        pre_x_valid_label = x_train0_label[validate_index]
        pre_y_valid = y_train0[validate_index]
        pre_x_train = x_train0[train_index]
        pre_x_train_label = x_train0_label[train_index]
        pre_y_train = y_train0[train_index]
        
        print 'COMPLETE \n '

        return  pre_x_train, pre_x_train_label, pre_x_test, pre_x_test_label, pre_x_valid,  \
                                pre_x_valid_label,  pre_y_train, pre_y_test, pre_y_valid

        
    def reshape_for_chainer_cnn(self, pre_x_train, pre_x_train_label, pre_x_test, pre_x_test_label, pre_x_valid,  \
                                pre_x_valid_label,  pre_y_train, pre_y_test, pre_y_valid): 
        
        ##### reshape x for cnn  #####
        print 'Reshaping the dataset... '

        
        x_train = np.zeros((pre_x_train.shape[0], 2,pre_x_train.shape[1],pre_x_train.shape[1]))
        x_train[:,0,:,:] = pre_x_train
        x_train[:,1,:,:] = pre_x_train_label       
  
        x_test = np.zeros((pre_x_test.shape[0], 2, pre_x_test.shape[1],pre_x_test.shape[1]))
        x_test[:,0,:,:] = pre_x_test
        x_test[:,1,:,:] = pre_x_test_label    

        x_valid = np.zeros((pre_x_valid.shape[0], 2, pre_x_valid.shape[1],pre_x_valid.shape[1]))
        x_valid[:,0,:,:] = pre_x_valid
        x_valid[:,1,:,:] = pre_x_valid_label       

        y_train = pre_y_train
        y_test = pre_y_test
        y_valid = pre_y_valid
        return x_train, x_test, x_valid, y_train, y_test, y_valid
        
    def filter_batch(self, batch, temperature):
        k = 0
        filtered_batch = batch.copy()
        for pic in batch:
            pic = ndi.filters.gaussian_filter(pic, 1.2)
            filtered_batch[k] = pic
            k = k +1
        return filtered_batch
        