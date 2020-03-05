######################## WiFi Locationing ###############

#### Libraries ####
if(require("pacman")=="FALSE"){
  install.packages("pacman")
}

#### Ignacio: This is to clean the user's environment
rm(list = ls(all = TRUE))
pacman::p_load("readr","ggplot2","dplyr","lubridate","plotly","scatterplot3d", "caret", "DMwR", "class", "e1071", 
               "caretEnsemble", "C50", "ISLR", "base", "randomForest", "stats", "ranger", "Metrics", "kknn", "gbm", 
               "ggpubr","gridExtra", "mlbench", "rayshader", "reshape2", "xgboost", "doParallel", "tidyverse","tidyr",
               "radiant.model", "plyr")

######################## Upload file and set seed ####
set.seed(123)

# Import Data
#### Ignacio: Emma, avoid making use of this approach as it is computer dependant
#### In other words, this will not work into another computer as the new computer
#### doesn't need to have the same OS and same folder tree.
#### You can use the function "list.files", and then use the function "grep"
#### to look for your script.

loc   <- grep("Wifi_fingerprint.R",list.files(recursive=TRUE),value=TRUE)
iloc  <- which(unlist(gregexpr("/Wifi_fingerprint.R$",loc)) != -1)
myloc <- paste(getwd(),loc[iloc],sep="/")
setwd(substr(myloc,1,nchar(myloc)-nchar("Wifi_fingerprint.R")))


#### Ignacio: Check if the file exists in the folder. However, this chunk
#### of code should be at the beggining of your script.
if (file.exists("All.Data.csv")) {
  ReadyData <-read_csv("All.Data.csv")
} else{
  # Load data
  Train <-read_csv("trainingData.csv")
  Valid <- read_csv("validationData.csv")
  
  #Distinct rows in order to only keep unique rows (rows with active WAP)
  Train <- dplyr::distinct(Train) # Observations: 19,937 to 19,300
  Valid <- dplyr::distinct(Valid) # Observations: 1,111 to 1,111
  
  #### Ignacio: Emma, you should justify the need of doing that.
  #Combinde train and valid data
  AllData <- rbind(Train, Valid)
  
  ################ Change columns into suiting data types ####
  AllData$BUILDINGID<-as.factor(AllData$BUILDINGID)
  AllData$FLOOR <- as.factor(AllData$FLOOR)
  
  #################################### Take away low activity WAPs (lower than 3%) #####
  k1 <- nearZeroVar(AllData[,1:520],uniqueCut = 0.03, saveMetrics = TRUE, allowParallel = TRUE)
  AllData <- AllData[,- which(k1$nzv==T)]
  #Observations: 20,411 
  #Variables: 400
  
  ############################ Delete variables that will not be used ####
  DeleteVar <- c("SPACEID", "RELATIVEPOSITION", "USERID", "PHONEID", "TIMESTAMP")
  for (i in DeleteVar) {AllData[,DeleteVar] <- NULL }
  
  write.csv(AllData, file = "All.Data.csv", row.names = F)
}

ReadyData <- AllData
#### Ignacio: Delete unneded dataframes to save memory
rm(AllData)

########################################################### Handle WAPs ####

#Split data in order to adjust WAPS 
Y_var<- ReadyData[c((ncol(ReadyData)-4):ncol(ReadyData))]
WAPs <- ReadyData[,1:391]

#Change to 100's to minus 100 
WAPs[WAPs == 100] <- -100

# Make all values positive and take away neagtive values
WAPs<- 100 + WAPs
WAPs[WAPs < 0] <- 0

#Take away WAPS with higher strenght than 70 due to not wanted in real life
WAPs[WAPs > 70 ] <- 0

#Plot new signal strenght
y <- stack(WAPs)
y <- y[-grep(0, y$values),]
hist(y$values, xlab = "WAP strength", main = "Distribution of WAPs signal stength", col = "blue", breaks=60)

#Create data frame
Data <-bind_cols(WAPs, Y_var)
Data$WAP5181<-NULL

############################################################################# Feature engineering
#Change to factor
Data$BUILDINGID <- as.factor (Data$BUILDINGID)
Data$FLOOR <- as.factor (Data$FLOOR)

##Create LOCATION variable
#### Ignacio: Emma, use this approach to prevent later problems. When you save
#### your data into a file, R doesn't save the variable type. Therefore, if
#### the separator is a "dot" when R reads back the file, will consider "0.0"
#### as "0", instead of a factor. 
#Data$LOCATION <- with(Data, interaction(BUILDINGID, FLOOR, drop = T))
Data$LOCATION <- paste(Data$BUILDINGID,Data$FLOOR,sep = "_")
Data$LOCATION <- as.factor(Data$LOCATION)

#Create variable with highest WAP vaule
new_data <- Data[,1:391] 

# row <- c()
# for (i in 1:nrow(new_data)){
#   a <- new_data[i, which.max(new_data[i,])]
#   row <- c(a, row)
#   print(i)
# }

#Add new column with max value of WAPs
#Data$MAX <- row
#### Ignacio: Much faster way
Data$MAX <- apply(new_data,1,which.max)

#### Ignacio: Min-max scaling
## Create simplyfied latitude and longitude
new_long <- Data$LONGITUDE - min(Data$LONGITUDE)
Data$LONGITUDE <- round(new_long, digits = 1)
new_lat <- Data$LATITUDE - min(Data$LATITUDE)
Data$LATITUDE <- round(new_lat, digits = 1)
longlat <- cbind(new_long, new_lat)

#### Save DF ready for models ####
if (!file.exists("Data_4_models.csv")) {
  write.csv(Data, file ="Data_4_models.csv", row.names = F)
} else{
  Model_data<-read.csv("Data_4_models.csv")
  
  #### Ignacio: Emma, free memory!!!
  rm(Data)
}

Model_data$LOCATION   <- as.factor(Model_data$LOCATION)
Model_data$BUILDINGID <- as.factor(Model_data$BUILDINGID)
Model_data$FLOOR      <- as.factor(Model_data$FLOOR)

############################################### Models #############################################

# Cluster <- makeCluster(5)
# registerDoParallel(Cluster)
################################################################################## BUILDING ####

########################################### Split data 
set.seed(123)
inTrain    <- createDataPartition(y = Model_data$BUILDINGID, p = 0.6,list = FALSE)
trainSet   <- Model_data[inTrain,]
test_valid <- Model_data[-inTrain,]

# smp_siz <- floor(0.5*nrow(test_valid))
# test_ind <- sample(seq_len(nrow(test_valid)), size = smp_siz) 

#### Ignacio: Emma, avoid making use of the "sample" function as it doesn't
#### respect the underlying distribution of your data.

test_ind <- createDataPartition(test_valid$BUILDINGID, p = 0.5, list = FALSE)

testSet  <- test_valid[test_ind,] 
validSet <- test_valid[-test_ind,]  

#########################################################################
# Check for "models" folder.
#### Ignacio: Emma, try to be more organized.
if (!dir.exists("models")) {
  dir.create(file.path(getwd(), "models"), showWarnings = FALSE)
  setwd(file.path(getwd(), "models"))
} else{ 
  setwd(file.path(getwd(), "models"))
}

################################################################ SVM ####

if (file.exists("SVM_Building_Model.rds")) {
  svm_building <- readRDS("SVM_Building_Model.rds")
} else{
  set.seed(123)
  system.time(svm_building <- svm(BUILDINGID ~ .-FLOOR -LOCATION -LONGITUDE -LATITUDE, data = trainSet))
  #Saving model
  saveRDS(svm_building,"SVM_Building_Model.rds")
}

#Output
svm_building

#Test the svm model
svm_building_pred_train <- predict(svm_building, newdata=trainSet)
svm_building_pred_test  <- predict(svm_building, newdata=testSet)
svm_building_pred_val   <- predict(svm_building, newdata=validSet)
  
# Confusion Matrix
print(svm_building_cm_train <- confusionMatrix(svm_building_pred_train, trainSet$BUILDINGID))
print(svm_building_cm_test  <- confusionMatrix(svm_building_pred_test,  testSet$BUILDINGID))
print(svm_building_cm_val   <- confusionMatrix(svm_building_pred_val,   validSet$BUILDINGID))

# Fill dataframe with a summary of results
Building <- data.frame(Model=rep("svm",3),
                        Set=c("Train","Test","Validation"),
                       Accuracy = c(round(svm_building_cm_train$overall[1],4),
                                    round(svm_building_cm_test$overall[1],4),
                                    round(svm_building_cm_val$overall[1],4)),
                       Kappa = c(round(svm_building_cm_train$overall[2],4),
                                 round(svm_building_cm_test$overall[2],4),
                                 round(svm_building_cm_val$overall[2],4)))

Building$Model <- factor(Building$Model,levels = c("svm","ranger","k-NN"))
Building$Set   <- factor(Building$Set,levels = c("Train","Test","Validation"))

################################################################ Ranger (Random Forest)

if (file.exists("Ranger_Building_Model.rds")) {
  ranger_building <- readRDS("Ranger_Building_Model.rds")
} else{
  set.seed(123)
  system.time(ranger_building<-ranger(BUILDINGID~.-LOCATION - FLOOR -LATITUDE - LONGITUDE, trainSet, importance="permutation"))
  #Saving model
  saveRDS(ranger_building,"Ranger_Building_Model.rds")
}

#Output
summary(ranger_building)
  
#Test the ranger model
ranger_building_pred_train  <- predict(ranger_building, trainSet)
ranger_building_pred_test   <- predict(ranger_building, testSet)
ranger_building_pred_val    <- predict(ranger_building, validSet)
  
# Confusion Matrix
ranger_building_cm_train <- confusionMatrix(ranger_building_pred_train$predictions,trainSet$BUILDINGID)
ranger_building_cm_test  <- confusionMatrix(ranger_building_pred_test$predictions,testSet$BUILDINGID)
ranger_building_cm_val   <- confusionMatrix(ranger_building_pred_val$predictions,validSet$BUILDINGID)

# Fill dataset with a summary of results
Building <- rbind(Building,c("ranger","Train",
                             round(ranger_building_cm_train$overall[1],4),
                             round(ranger_building_cm_train$overall[2],4)))
Building <- rbind(Building,c("ranger","Test",
                             round(ranger_building_cm_test$overall[1],4),
                             round(ranger_building_cm_test$overall[2],4)))
Building <- rbind(Building,c("ranger","Validation",
                             round(ranger_building_cm_val$overall[1],4),
                             round(ranger_building_cm_val$overall[2],4)))

Building$Accuracy <- as.numeric(Building$Accuracy)
Building$Kappa    <- as.numeric(Building$Kappa)

ggplot(melt(Building), aes(x=Model, y=value,fill=Model))+
  geom_col()+
  facet_grid(variable~Set, scales="free") +
  theme(legend.position = "none") +
  ggtitle("Model error metrics in predicting Building")

################################################################################## FLOOR #####

######################################### Split data 
set.seed(123)
inTrainFloor <- createDataPartition(y = Model_data$LOCATION, p = 0.6, list = FALSE)
trainSet     <- Model_data [inTrainFloor,]
test_valid   <- Model_data[-inTrainFloor,]

#### Ignacio: Emma, avoid making use of the "sample" function as it doesn't
#### respect the underlying distribution of your data.

set.seed(123)
test_ind <- createDataPartition(test_valid$LOCATION, p = 0.5, list = FALSE)

# set.seed(123)
# smp_siz = floor(0.5*nrow(test_valid))
# test_ind = sample(seq_len(nrow(test_valid)), size = smp_siz) 

testSet  <- test_valid[test_ind,] 
validSet <- test_valid[-test_ind,] 

########################################################################### SVM ####

if (file.exists("SVM_Floor_Model.rds")) {
  svm_floor <- readRDS("SVM_Floor_Model.rds")
} else{
  set.seed(123)
  system.time(svm_floor <- svm(LOCATION ~ .-BUILDINGID - FLOOR - LOCATION - LATITUDE , data = trainSet))
  #Saving model
  saveRDS(svm_floor,"SVM_Floor_Model.rds")
} 

#Output
summary(svm_floor)
  
#Test the svm model
svm_floor_pred_train  <- predict(svm_floor, newdata=trainSet)
svm_floor_pred_test   <- predict(svm_floor, newdata=testSet)
svm_floor_pred_val    <- predict(svm_floor, newdata=validSet)
  
#Confusion Matrix
svm_floor_cm_train <- confusionMatrix(svm_floor_pred_train,trainSet$LOCATION)
svm_floor_cm_test  <- confusionMatrix(svm_floor_pred_test,testSet$LOCATION)
svm_floor_cm_val   <- confusionMatrix(svm_floor_pred_val,validSet$LOCATION)

# Fill dataframe with a summary of results
Floor <- data.frame(Model=rep("svm",3),
                       Set=c("Train","Test","Validation"),
                       Accuracy = c(round(svm_floor_cm_train$overall[1],4),
                                    round(svm_floor_cm_test$overall[1],4),
                                    round(svm_floor_cm_val$overall[1],4)),
                       Kappa = c(round(svm_floor_cm_train$overall[2],4),
                                 round(svm_floor_cm_test$overall[2],4),
                                 round(svm_floor_cm_val$overall[2],4)))

Floor$Model <- factor(Floor$Model,levels = c("svm","ranger","k-NN"))
Floor$Set   <- factor(Floor$Set,levels = c("Train","Test","Validation"))

######################################################## Ranger (Random Forest) ####

if (file.exists("Ranger_Floor_Model.rds")) {
  ranger_floor<- readRDS("Ranger_Floor_Model.rds")
} else{
  set.seed(123)
  system.time(ranger_floor<-ranger(LOCATION~.-BUILDINGID - FLOOR -LATITUDE - LONGITUDE, trainSet, importance="permutation"))
  #Saving model
  saveRDS(ranger_floor,"Ranger_Floor_Model.rds")
} 

#Output
summary(ranger_floor)
  
#Test the ranger model
ranger_floor_pred_train   <- predict(ranger_floor, trainSet)
ranger_floor_pred_test    <- predict(ranger_floor, testSet)
ranger_floor_pred_valid   <- predict(ranger_floor, validSet)
  
#Confusion Matrix
ranger_floor_cm_train <- confusionMatrix(ranger_floor_pred_train$predictions,trainSet$LOCATION)
ranger_floor_cm_test  <- confusionMatrix(ranger_floor_pred_test$predictions,testSet$LOCATION)
ranger_floor_cm_val   <- confusionMatrix(ranger_floor_pred_valid$predictions,validSet$LOCATION)

# Fill dataset with a summary of results
Floor <- rbind(Floor,c("ranger","Train",
                             round(ranger_floor_cm_train$overall[1],4),
                             round(ranger_floor_cm_train$overall[2],4)))
Floor <- rbind(Floor,c("ranger","Test",
                             round(ranger_floor_cm_test$overall[1],4),
                             round(ranger_floor_cm_test$overall[2],4)))
Floor <- rbind(Floor,c("ranger","Validation",
                             round(ranger_floor_cm_val$overall[1],4),
                             round(ranger_floor_cm_val$overall[2],4)))

Floor$Accuracy <- as.numeric(Floor$Accuracy)
Floor$Kappa    <- as.numeric(Floor$Kappa)

ggplot(melt(Floor), aes(x=Model, y=value,fill=Model))+
  geom_col()+
  facet_grid(variable~Set, scales="free") +
  theme(legend.position = "none") +
  ggtitle("Model error metrics in predicting Floor")

  
#separate column
#extract(data, col, into, regex = "([[:alnum:]]+)", remove = TRUE,
#        convert = FALSE, ...)
  
#If you just want the second variable:
df <- data.frame(ranger_floor_pred = c("LOCATION"))
df %>% separate("LOCATION", c("NA", "Predicted Floor"))
  
#PROV <-ranger_floor_pred %>% separate(LOCATION, c(NA, "Predicted floor"))
PROV <- as.factor(substr(ranger_floor_pred$predictions,3,3))


############################################################################## LATITUDE ####

#Functions to imit errors in predicitons with straight line equation
below <- function(x,pred) {value = (x * -.51) + 175; if(value > pred){return(value)}; return(pred)}
above <- function(x,pred) {value = (x * -0.56) + 315; if(value > pred) {return(value)}; return(pred)}


######################################### Split data 
set.seed(123)
inTrainFloor<- createDataPartition(y = Model_data$LATITUDE, p = 0.6,list = FALSE)
trainSet   <- Model_data [inTrainFloor,]
test_valid <- Model_data[-inTrainFloor,]

test_ind <- createDataPartition(test_valid$LATITUDE, p = 0.5, list = FALSE)
#smp_siz = floor(0.5*nrow(test_valid))
#test_ind = sample(seq_len(nrow(test_valid)), size = smp_siz) 

testSet  <- test_valid[test_ind,] 
validSet <- test_valid[-test_ind,]

############################################################################# SVM ####

if (file.exists("SVM_Latitude_Model.rds")) {
  svm_lat <- readRDS("SVM_Latitude_Model.rds")
} else{
  set.seed(123)
  system.time(svm_lat <- svm(LATITUDE ~ .+MAX -BUILDINGID - FLOOR - LOCATION - LONGITUDE, data = trainSet))
  #Saving model
  saveRDS(svm_lat,"SVM_Latitude_Model.rds")
}

#Output
summary(svm_lat)
  
#Predict the svm model
svm_lat_pred_train  <- predict(svm_lat, newdata=trainSet)
svm_lat_pred_test   <- predict(svm_lat, newdata=testSet)
svm_lat_pred_val    <- predict(svm_lat, newdata=validSet)
  
#Error metrics
svm_lat_err_train <- postResample(svm_lat_pred_train,trainSet$LATITUDE)
svm_lat_err_test  <- postResample(svm_lat_pred_test,testSet$LATITUDE)
svm_lat_err_val   <- postResample(svm_lat_pred_val,validSet$LATITUDE)

#Fill dataset with a summary of results
Latitude <- data.frame(Model=rep("svm",3),
                       Set=c("Train","Test","Validation"),
                       RMSE = c(round(svm_lat_err_train[1],2),
                                round(svm_lat_err_test[1],2),
                                round(svm_lat_err_val[1],2)),
                       Rsquared = c(round(svm_lat_err_train[2],2),
                                    round(svm_lat_err_test[2],2),
                                    round(svm_lat_err_val[2],2)),
                       MAE = c(round(svm_lat_err_train[3],2),
                               round(svm_lat_err_test[3],2),
                               round(svm_lat_err_val[3],2)))

Latitude$Model <- factor(Latitude$Model,levels = c("svm","ranger","k-NN"))
Latitude$Set   <- factor(Latitude$Set,levels = c("Train","Test","Validation"))

########################################################### Ranger (Random forest) ####

if (file.exists("Ranger_Latitude_Model.rds")) {
  ranger_lat <- readRDS("Ranger_Latitude_Model.rds")
} else{
  set.seed(123)
  system.time(ranger_lat <- ranger(LATITUDE ~. -BUILDINGID - FLOOR - LOCATION - LATITUDE ,trainSet,  importance = "permutation", case.weights = T))

  #Saving model
  saveRDS(ranger_lat,"Ranger_Latitude_Model.rds")
  
} 

#Output
summary(ranger_lat)
  
#Predict the ranger model
ranger_lat_pred_train  <- predict(ranger_lat,trainSet)
ranger_lat_pred_test   <- predict(ranger_lat,testSet)
ranger_lat_pred_val    <- predict(ranger_lat,validSet)
  
#Error metrics
ranger_lat_err_train <- postResample(ranger_lat_pred_train$predictions,trainSet$LATITUDE)
ranger_lat_err_test  <- postResample(ranger_lat_pred_test$predictions,testSet$LATITUDE)
ranger_lat_err_val   <- postResample(ranger_lat_pred_val$predictions,validSet$LATITUDE)

#Update dataset with a summary of results
Latitude <- rbind(Latitude,c("ranger","Train",round(ranger_lat_err_train,2)))
Latitude <- rbind(Latitude,c("ranger","Test",round(ranger_lat_err_test,2)))
Latitude <- rbind(Latitude,c("ranger","Validation",round(ranger_lat_err_val,2)))

################################################################################ KNN ####

if (file.exists("knn_Latitude_Model.rds")) {
  knn_lat <- readRDS("knn_Latitude_Model.rds")
} else{
  set.seed(123)
  system.time(knn_lat <-knnreg(LATITUDE~.,data = trainSet[, -which(names(trainSet) %in% c("LONGITUDE","BUILDINGID",
                                                                                          "FLOOR", "LOCATION"))]))

  #Saving model
  saveRDS(knn_lat,"knn_Latitude_Model.rds")
  
}

#Test model
knn_lat_pred_train <- predict(knn_lat,trainSet)
knn_lat_pred_test  <- predict(knn_lat,testSet)
knn_lat_pred_val   <- predict(knn_lat,validSet)
  
#Metrics
knn_lat_err_train <- postResample(knn_lat_pred_train,trainSet$LATITUDE)
knn_lat_err_test  <- postResample(knn_lat_pred_test,testSet$LATITUDE)
knn_lat_err_val   <- postResample(knn_lat_pred_val,validSet$LATITUDE)

#Update dataset with a summary of results
Latitude <- rbind(Latitude,c("k-NN","Train",round(knn_lat_err_train,2)))
Latitude <- rbind(Latitude,c("k-NN","Test",round(knn_lat_err_test,2)))
Latitude <- rbind(Latitude,c("k-NN","Validation",round(knn_lat_err_val,2)))

Latitude$RMSE     <- as.numeric(Latitude$RMSE)
Latitude$Rsquared <- as.numeric(Latitude$Rsquared)
Latitude$MAE      <- as.numeric(Latitude$MAE)

ggplot(melt(Latitude), aes(x=Model, y=value,fill=Model))+
  geom_col()+
  facet_grid(variable~Set, scales="free") +
  theme(legend.position = "none") +
  ggtitle("Model error metrics in predicting Latitude")


##################################################################################### LONGITUDE ####

######################################### Split data into training and testing set
####
#### Ignacio: Emma, you can't split again the dataset as Longitude and Latitude are tied. 
# set.seed(123)
# inTrainFloor<- createDataPartition(y = Model_data$LONGITUDE, p = 0.6,list = FALSE)
# trainSet <- Model_data [inTrainFloor,]
# test_valid<- Model_data[-inTrainFloor,]
# 
# # set.seed(123)
# # smp_siz = floor(0.5*nrow(test_valid))
# # test_ind = sample(seq_len(nrow(test_valid)), size = smp_siz) 
# 
# test_ind <- createDataPartition(y = test_valid$LONGITUDE, p = 0.5,list = FALSE)
# testSet  <- test_valid[test_ind,] 
# validSet <- test_valid[-test_ind,]  


################################################################################## SVM ####

if (file.exists("SVM_Longitude_Model.rds")) {
  svm_long <- readRDS("SVM_Longitude_Model.rds")
} else{
  set.seed(123)
  system.time(svm_long <- svm(LONGITUDE ~ . -BUILDINGID - FLOOR - LOCATION - LATITUDE, data = trainSet))

  #Saving model
  saveRDS(svm_long,"SVM_Longitude_Model.rds")
  
} 

#Output
summary(svm_long)
  
#Predict the svm model
svm_long_pred_train <- predict(svm_long, newdata=trainSet)
svm_long_pred_test  <- predict(svm_long, newdata=testSet)
svm_long_pred_val   <- predict(svm_long, newdata=validSet)
  
#Metrics
svm_long_err_train <- postResample(svm_long_pred_train,trainSet$LONGITUDE)
svm_long_err_test  <- postResample(svm_long_pred_test,testSet$LONGITUDE)
svm_long_err_val   <- postResample(svm_long_pred_val,validSet$LONGITUDE)

#Fill dataset with a summary of results
Longitude <- data.frame(Model=rep("svm",3),
                       Set=c("Train","Test","Validation"),
                       RMSE = c(round(svm_long_err_train[1],2),
                                round(svm_long_err_test[1],2),
                                round(svm_long_err_val[1],2)),
                       Rsquared = c(round(svm_long_err_train[2],2),
                                    round(svm_long_err_test[2],2),
                                    round(svm_long_err_val[2],2)),
                       MAE = c(round(svm_long_err_train[3],2),
                               round(svm_long_err_test[3],2),
                               round(svm_long_err_val[3],2)))

Longitude$Model <- factor(Longitude$Model,levels = c("svm","ranger","k-NN"))
Longitude$Set   <- factor(Longitude$Set,levels = c("Train","Test","Validation"))

########################################################################## Ranger (Random Forest)

if (file.exists("Ranger_Longitude_Model.rds")) {
  ranger_long <- readRDS("Ranger_Longitude_Model.rds")
} else{
  set.seed(123)
  system.time(ranger_long <- ranger(LONGITUDE ~ .-BUILDINGID - FLOOR - LOCATION - LATITUDE, trainSet, importance = "permutation", case.weights = T))

  #Save Ranger Longitude model
  saveRDS(ranger_long,"Ranger_Longitude_Model.rds")
  
}

#Output
summary(ranger_long)
  
#Test the ranger model
ranger_long_pred_train <- predict(ranger_long,trainSet)
ranger_long_pred_test  <- predict(ranger_long,testSet)
ranger_long_pred_val   <- predict(ranger_long,validSet)
  
#Metrics
ranger_long_err_train <- postResample(ranger_long_pred_train$predictions,trainSet$LONGITUDE)
ranger_long_err_test  <- postResample(ranger_long_pred_test$predictions, testSet$LONGITUDE)
ranger_long_err_val   <- postResample(ranger_long_pred_val$predictions, validSet$LONGITUDE)

#Update dataset with a summary of results
Longitude <- rbind(Longitude,c("ranger","Train",round(ranger_long_err_train,2)))
Longitude <- rbind(Longitude,c("ranger","Test",round(ranger_long_err_test,2)))
Longitude <- rbind(Longitude,c("ranger","Validation",round(ranger_long_err_val,2)))

###################################################################################### KNN ####

if (file.exists("knn_Longitude_Model.rds")) {
  knn_long <- readRDS("knn_Longitude_Model.rds")
} else{
  set.seed(123)
  system.time(knn_long <-knnreg(LONGITUDE~.,data = trainSet[, -which(names(trainSet) %in% c("LATITUDE","BUILDINGID",
                                                                                            "FLOOR", "LOCATION"))]))
  #Saving model
  saveRDS(knn_long,"knn_Longitude_Model.rds")
  
}
  
#Test model
knn_long_pred_train <- predict(knn_long,trainSet)
knn_long_pred_test  <- predict(knn_long,testSet)
knn_long_pred_val   <- predict(knn_long,validSet)
  
#Metrics
knn_long_err_train <- postResample(knn_long_pred_train,trainSet$LONGITUDE)
knn_long_err_test  <- postResample(knn_long_pred_test,testSet$LONGITUDE)
knn_long_err_val   <- postResample(knn_long_pred_val,validSet$LONGITUDE)

#Update error metrics dataset
Longitude <- rbind(Longitude,c("k-NN","Train",round(knn_long_err_train,2)))
Longitude <- rbind(Longitude,c("k-NN","Test",round(knn_long_err_test,2)))
Longitude <- rbind(Longitude,c("k-NN","Validation",round(knn_long_err_val,2)))

Longitude$RMSE     <- as.numeric(Longitude$RMSE)
Longitude$Rsquared <- as.numeric(Longitude$Rsquared)
Longitude$MAE      <- as.numeric(Longitude$MAE)

ggplot(melt(Longitude), aes(x=Model, y=value,fill=Model))+
  geom_col()+
  facet_grid(variable~Set, scales="free") +
  theme(legend.position = "none") +
  ggtitle("Model error metrics in predicting Latitude")

######################################################################## Errors #####################
#Data frame to plot & compare longitude & latitude with real location
#### Ignacio: Emma, you can't do this because for Longitude, you create one specific dataset 
#### split in: Train, Test, Val, which was DIFFERENT fron the one you created for Latitude.
#### And what's more, when you did the splits, you overwrite the datasets so you can do this
#### as the results are totally messup.
# Lat_Long_Diff <- data.frame(
#   LONG.RANGER <- ranger_long_pred_val$predictions,
#   LONG.KNN <-knn_long_pred_val,
#   LONG.SVM <- svm_long_pred_val,
#   LAT.SVM <- svm_lat_pred_val,
#   LAT.RANGER<- ranger_lat_pred_val$predictions,
#   LAT.KNN <- knn_lat_pred_val,
#   LATITUDE <- validSet$LATITUDE,
#   LONGITUDE <- validSet$LONGITUDE,
#   BUILDING = validSet$BUILDINGID,
#   FLOOR = validSet$FLOOR)
  
####################################################################### LATITUDE ERRORS ####
# #Only latitude
#   plot_ly(Lat_Long_Diff, x = ~LONGITUDE, y = ~LATITUDE, type = "scatter", name = "Real location") %>%
#     add_trace(Lat_Long_Diff, y = ~LAT.SVM, name = "SVM prediction")%>%
#     add_trace(Lat_Long_Diff,y = ~LAT.RANGER,name = "Ranger prediction") %>%
#     add_trace(Lat_Long_Diff, x = ~LONG.KNN, name = "KNN prediction") %>%
#     layout(title = "Predicted latitude")
# 
#   
# #DF to plot latitude errors
# Diff.lat.knn<- testSet$LATITUDE - knn_lat_pred
# Diff.lat.Ranger <- testSet$LATITUDE - ranger_lat_pred$predictions 
# Diff.lat.svm <- testSet$LATITUDE - svm_lat_pred
#   
# Z <- data.frame(v1=Diff.lat.knn, v2=Diff.lat.Ranger, v3=Diff.lat.svm)
# colnames(Z) <- c("KNN","RANGER", "SVM")
# Z<-melt(Z)
#   
# #Errors latitude 
#   ggplot(Z,aes(x=value,fill=variable))  +
#     labs(fill="") +
#     geom_density(alpha=0.2) +
#     ggtitle("Error distribution latitude") +
#     theme_void()
#   
#   
# ###################################################################### LONGITUDE ERRORS ####  
# #Only longitude
#   plot_ly(Lat_Long_Diff, x = ~LONGITUDE, y = ~LATITUDE, type = "scatter", name = "Real location") %>%
#     add_trace(Lat_Long_Diff, x = ~LONG.SVM, name = "SVM prediction") %>%
#     add_trace(Lat_Long_Diff, x = ~LONG.KNN, name = "KNN prediction") %>%
#     add_trace(Lat_Long_Diff,x = ~LONG.RANGER,name = "Ranger prediction")%>%
#     layout(title = "Predicted longitude")
#   
# #DF to plot longitude errors
# Diff.long.knn    <- testSet$LONGITUDE - knn_long_pred
# Diff.long.Ranger <- testSet$LONGITUDE - ranger_long_pred$predictions 
# Diff.long.svm    <- testSet$LONGITUDE - svm_long_pred 
# 
# mean(Diff.long.knn)
# mean(Diff.long.Ranger)
# mean(Diff.long.svm )
# mean(abs(Diff.long.knn))
# 
# G <- data.frame(v1=Diff.long.knn, v2=Diff.long.Ranger, v3=Diff.long.svm )
# colnames(G) <- c("KNN","RANGER","SVM")
# G<-melt(G)
# 
# H <- data.frame(v1=Diff.long.Ranger, v2=Diff.long.svm)
# colnames(H) <- c("RANGER","SVM")
# H <-melt(H)
# 
# # Plot errors longitude
#   ggplot(H,aes(x=value,fill=variable))  +
#     labs(fill="") +
#     geom_density(alpha=0.2) +
#     ggtitle("Error distribution longitude") +
#     theme_void()

#### TO DO: Emma, you were re-using a lot of pieces of code in your script. 
#### Try to put repetitive process inside functions in order to clean the
#### code and making easier to mantain.
####
#### Error analysis: It has been quite poor. One of the objectives of this
#### task was to go beyond the error metrics. Where each model is making 
#### misclassifications for Building and Floor? Are all the models making
#### mistakes in the same regions? Try to use plotly in order to make scatter
#### plots color coded by model and error.
#### In case of regression, compute the Ecluidian error in each location 
#### and use a plotly bubble plot in order to see where each model is making
#### bigger mistakes. Code the model by color. Can you see areas in which 
#### all the models makes big mistakes? If so, what this is telling you?