import train_cnn_script as tc

cropsize = 25
resolution = 256
epochsize = 80
cooling_rate = 0.95
initial_temperature = 1.5
boringsize = 7 



import train_cnn_conditional as tc
mycnn_cdd = tc.train_cnn_cdd(cropsize = cropsize, resolution = resolution)
mycnn_cdd.train(epochsize = epochsize, cooling_rate = cooling_rate, \
                initial_temperature= initial_temperature, boringsize = boringsize) 
