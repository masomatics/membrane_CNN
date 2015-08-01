import train_cnn_script as tc

epoch = 100
#mycnn = tc.train_cnn(crop = 33, resol= 256)
#mycnn.train(epoch)


import train_cnn_conditional as tc
mycnn_cdd = tc.train_cnn_cdd(crop = 15, resol= 256)
mycnn_cdd.train(epoch)