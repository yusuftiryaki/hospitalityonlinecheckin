# Load the H2O library and start up the H2O cluster locally on your machine
library(h2o)
h2o.init(nthreads = 1, #Number of threads -1 means use all cores on your machine
         max_mem_size = "8G")  #max mem size is the maximum memory to allocate to H2O



source("HospitalityOnlineCheckIn_DataPrepparation.R")
checkindata$GuestId <- as.factor(checkindata$GuestId)
checkindata$EmailDomain <- as.factor(checkindata$EmailDomain)
checkindata$CountryCode <- as.factor(checkindata$CountryCode)
checkindata$RoomSelected <- as.factor(checkindata$RoomSelected)


data <- as.h2o(checkindata)   
dim(data)

# Since we want to train a binary classification model, 
# we must ensure that the response is coded as a factor
# If the response is 0/1, H2O will assume it's numeric,
# which means that H2O will train a regression model instead
data$RoomSelected <- as.factor(data$RoomSelected)  #encode the binary repsonse as a factor
h2o.levels(data$RoomSelected)  #optional: after encoding, this shows the two factor levels, '0' and '1'
# [1] "0" "1"

# Partition the data into training, validation and test sets
splits <- h2o.splitFrame(data = data, 
                         ratios = c(0.8, 0.05),  #partition data into 70%, 15%, 15% chunks
                         seed = 1)  #setting a seed will guarantee reproducibility
train <- splits[[1]]
valid <- splits[[2]]
test <- splits[[3]]

# Take a look at the size of each partition
# Notice that h2o.splitFrame uses approximate splitting not exact splitting (for efficiency)
# so these are not exactly 70%, 15% and 15% of the total rows
nrow(train) 
nrow(valid) 
nrow(test) 

# Identify response and predictor variables
y <- "RoomSelected"
x <- setdiff(names(data), c(y))  #remove the  outcome
print(x)


rf_fit <- h2o.randomForest(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "rf_fit3",
                            mtries = 8,
                            ntrees = 100,
                            seed = 1
                            )

# To evaluate the cross-validated AUC, do the following:
#h2o.auc(rf_fit3, xval = TRUE)  
rf_perf <- h2o.performance(model = rf_fit,
                            newdata = test)
h2o.auc(rf_fit, train = T)
h2o.auc(rf_perf) 
plot(rf_fit, 
     timestep = "number_of_trees", 
     metric = "AUC")
h2o.varimp_plot(rf_fit)


# Train a DL with new architecture and more epochs.
# Next we will increase the number of epochs used in the GBM by setting `epochs=20` (the default is 10).  
# Increasing the number of epochs in a deep neural net may increase performance of the model, however, 
# you have to be careful not to overfit your model to your training data.  To automatically find the optimal number of epochs, 
# you must use H2O's early stopping functionality.  Unlike the rest of the H2O algorithms, H2O's DL will 
# use early stopping by default, so for comparison we will first turn off early stopping.  We do this in the next example 
# by setting `stopping_rounds=0`.
dl_fit <- h2o.deeplearning(x = x,
                            y = y,
                            training_frame = train,
                            model_id = "dl_fit",
                            validation_frame = valid,  #only used if stopping_rounds > 0
                            epochs = 50,
                            hidden= c(18,48,66,84,102,120,138,156), #c(48,66,84,102,120,138,156),
                            stopping_rounds = 3,  # set 0 to disable early stopping
                            seed = 1,
                            reproducible = TRUE,
                            #l1 = 0.0001,
                            l2 = 0.0005)

dl_perf <- h2o.performance(model = dl_fit,
                            newdata = test)
h2o.auc(dl_fit, train = T)
h2o.auc(dl_perf)
# Look at scoring history for  DL model
plot(dl_fit, 
     timestep = "epochs", 
     metric = "AUC")
h2o.varimp_plot(dl_fit)

#check some samples
sample_index <- sort(sample(1:nrow(test),100, replace = F))
p_rf <- h2o.predict(rf_fit, test[sample_index,])
p_dl <- h2o.predict(dl_fit, test[sample_index,])
random_Samples <- h2o.cbind(p_rf, p_dl,test[sample_index,])
View(random_Samples)



