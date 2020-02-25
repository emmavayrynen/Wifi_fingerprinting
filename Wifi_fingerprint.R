######################## WiFi Locationing ###############

#### Libraries ####
if(require("pacman")=="FALSE"){
  install.packages("pacman")
}

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
  #Distinct rows in order to only keep unique rows (rows with active WAP)
  Train <- dplyr::distinct(Train) # Observations: 19,937 to 19,300
  Valid <- dplyr::distinct(Valid) # Observations: 1,111 to 1,111
  
  #### Ignacio: Emma, you should justify the need of doing that.
  #Combinde train and valid data
  AllData <- rbind (Train, Valid)
  
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
Data$LOCATION <- with(Data, interaction(BUILDINGID, FLOOR, drop = T))
Data$LOCATION <- as.factor (Data$LOCATION)

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
write.csv(Data, file ="Data_4_models.csv", row.names = F)
Model_data<-read.csv("Data_4_models.csv")

Model_data$LOCATION <-as.factor(Model_data$LOCATION)
Model_data$BUILDINGID <-as.factor(Model_data$BUILDINGID)
Model_data$FLOOR <-as.factor(Model_data$FLOOR)
############################################### Models #############################################

Cluster <- makeCluster(5)
registerDoParallel(Cluster)
################################################################################## BUILDING ####

########################################### Split data 
set.seed(123)
inTrain    <- createDataPartition(y = Model_data$BUILDINGID, p = 0.6, t=3,list = FALSE)
trainSet   <- Model_data[inTrain,]
test_valid <- Model_data[-inTrain,]

# smp_siz <- floor(0.5*nrow(test_valid))
# test_ind <- sample(seq_len(nrow(test_valid)), size = smp_siz) 

#### Ignacio: Emma, avoid making use of the "sample" function as it doesn't
#### respect the underlying distribution of your data.

test_ind <- createDataPartition(test_valid$BUILDINGID, p = 0.5, list = FALSE)

testSet  <- test_valid[test_ind,] 
validSet <- test_valid[-test_ind,]  

################################################################ SVM ####
set.seed(123)
system.time(svm_building <- svm(BUILDINGID ~ .-FLOOR -LOCATION -LONGITUDE -LATITUDE, data = trainSet))

#Output
svm_building

#Test the svm model
svm_building_pred  <- predict(svm_building, newdata=testSet)
svm_building_predV <- predict(svm_building, newdata=validSet)

# Confusion Matrix
print(svm_building_cm <- confusionMatrix (svm_building_pred, testSet$BUILDINGID))
print(svm_building_cmV <- confusionMatrix (svm_building_predV, validSet$BUILDINGID))

#Save model
#### Ignacio: Emma, try to be more organized.
dir.create(file.path(getwd(), "models"), showWarnings = FALSE)
setwd(file.path(getwd(), "models"))

saveRDS(svm_building,"SVM_Building_Model.rds")

################################################################ Ranger (Random Forest)
set.seed(123)
system.time(ranger_building<-ranger(BUILDINGID~.-LOCATION - FLOOR -LATITUDE - LONGITUDE, trainSet, importance="permutation"))

#Output
summary(ranger_building)

#Test the ranger model
ranger_building_pred <- predict(ranger_building, testSet)
ranger_building_predV <- predict(ranger_building, validSet)

# Confusion Matrix
ranger_table_building<-table(testSet$BUILDINGID, ranger_building_pred$predictions)
print(ranger_building_cm<-(confusionMatrix(ranger_table_building)))

#ranger_table_buildingV<-table(validSet$BUILDINGID, ranger_building_predV$predictions)
ranger_buildingV_cmV <- confusionMatrix(ranger_building_predV$predictions,validSet$BUILDINGID)
print(ranger_buildingV_cmV)

#saveRDS(ranger_building,"C:/Users/46768/Documents/Dota/Wi-Fi position/Ranger_Building_Model.rds")
saveRDS(ranger_building,"Ranger_Building_Model.rds")

################################################################################## FLOOR #####

######################################### Split data 
set.seed(123)
inTrainFloor <- createDataPartition(y = Model_data$LOCATION, p = 0.6, t=3, list = FALSE)
trainSet     <- Model_data [inTrainFloor,]
test_valid   <- Model_data[-inTrainFloor,]

#### Ignacio: Emma, avoid making use of the "sample" function as it doesn't
#### respect the underlying distribution of your data.

set.seed(123)
test_ind <- createDataPartition(test_valid$LOCATION, p = 0.5, list = FALSE)

# set.seed(123)
# smp_siz = floor(0.5*nrow(test_valid))
# test_ind = sample(seq_len(nrow(test_valid)), size = smp_siz) 

testSet = test_valid[test_ind,] 
validSet =test_valid[-test_ind,]  
########################################################################### SVM ####
set.seed(123)
system.time(svm_floor <- svm(LOCATION ~ .-BUILDINGID - FLOOR - LOCATION - LATITUDE , data = trainSet))

#Output
summary(svm_floor)

#Test the svm model
svm_floor_pred <- predict(svm_floor, newdata=testSet)
svm_floor_predV <- predict(svm_floor, newdata=validSet)

#Confusion Matrix
print(svm_floor_cm <- confusionMatrix (svm_floor_pred, testSet$LOCATION))
svm_floor_cmV <- confusionMatrix (svm_floor_predV, validSet$LOCATION)

#Save Decision tree floor model
saveRDS(svm_floor,"C:/Users/46768/Documents/Dota/Wi-Fi position/SVM_Floor_Model.rds")

######################################################## Ranger (Random Forest) ####
set.seed(123)
system.time(ranger_floor<-ranger(LOCATION~.-BUILDINGID - FLOOR -LATITUDE - LONGITUDE, trainSet, importance="permutation"))

#Output
summary(ranger_floor)

#Test the ranger model
ranger_floor_pred  <- predict(ranger_floor, testSet)
ranger_floor_predV <- predict(ranger_floor, validSet)

#Confusion Matrix
ranger_table_floor<-table(testSet$LOCATION, ranger_floor_pred$predictions)
print(ranger_floor_cm<-(confusionMatrix(ranger_table_floor)))
ranger_table_floorV<-table(validSet$LOCATION, ranger_floor_predV$predictions)
print(ranger_floor_cmV<-(confusionMatrix(ranger_table_floorV)))

#seperate column
extract(data, col, into, regex = "([[:alnum:]]+)", remove = TRUE,
        convert = FALSE, ...)

#If you just want the second variable:
df <- data.frame(ranger_floor_pred = c("LOCATION"))
df %>% separate("LOCATION", c("NA", "Predicted Floor"))

PROV <-ranger_floor_pred %>% separate(LOCATION, c(NA, "Predicted floor"))

#Save Ranger floor model
saveRDS(ranger_floor,"C:/Users/46768/Documents/Dota/Wi-Fi position/Ranger_Floor_Model.rds")

############################################################################## LATITUDE ####

#Functions to imit errors in predicitons with straight line equation
below <- function(x,pred) {value = (x * -.51) + 175; if(value > pred){return(value)}; return(pred)}
above <- function(x,pred) {value = (x * -0.56) + 315; if(value > pred) {return(value)}; return(pred)}


######################################### Split data 
set.seed(123)
inTrainFloor<- createDataPartition(y = Model_data$LATITUDE, p = 0.6, t=3,list = FALSE)
trainSet <- Model_data [inTrainFloor,]
test_valid<- Model_data[-inTrainFloor,]

set.seed(123)
smp_siz = floor(0.5*nrow(test_valid))
test_ind = sample(seq_len(nrow(test_valid)), size = smp_siz) 

testSet = test_valid[test_ind,] 
validSet =test_valid[-test_ind,]

############################################################################# SVM ####
set.seed(123)
system.time(svm_lat <- svm(LATITUDE ~ .+MAX -BUILDINGID - FLOOR - LOCATION - LONGITUDE, data = trainSet))

#Output
summary(svm_lat)

#Predict the svm model
svm_lat_pred  <- predict(svm_lat, newdata=testSet)
svm_lat_predV <- predict(svm_lat, newdata=validSet)


#Error metrics
print(postResample(svm_lat_pred, testSet$LATITUDE))
print(postResample(svm_lat_predV,validSet$LATITUDE))


#Save
saveRDS(svm_lat,"C:/Users/46768/Documents/Dota/Wi-Fi position/SVM_Latitude_Model.rds")

########################################################### Ranger (Random forest) ####
set.seed(123)
system.time(ranger_lat <- ranger(LATITUDE ~. -BUILDINGID - FLOOR - LOCATION - LATITUDE ,trainSet,  importance = "permutation", case.weights = T))

#Output
summary(ranger_lat)

#Predict the ranger model
ranger_lat_pred  <-  predict(ranger_lat, testSet)
ranger_lat_predV <- predict(ranger_lat, validSet)

#Error metrics
print(postResample(ranger_lat_pred$predictions, testSet$LATITUDE))
print(postResample(ranger_lat_predV$predictions, validSet$LATITUDE)) 

#Save model
saveRDS(ranger_lat,"C:/Users/46768/Documents/Dota/Wi-Fi position/Ranger_Latitude_Model.rds")

################################################################################ KNN ####
set.seed(123)
system.time(knn_lat <-knnreg(LATITUDE~.,data = trainSet[, -which(names(trainSet) %in% c("LONGITUDE","BUILDINGID",
                                                                                          "FLOOR", "LOCATION"))]))
#Test model
knn_lat_pred  <- predict(knn_lat, testSet)
knn_lat_predV <- predict(knn_lat, validSet)

#Metrics
print(postResample(knn_lat_pred, testSet$LATITUDE))
print(postResample(knn_lat_predV, validSet$LATITUDE))

##################################################################################### LONGITUDE ####

######################################### Split data into training and testing set
set.seed(123)
inTrainFloor<- createDataPartition(y = Model_data$LONGITUDE, p = 0.6, t = 3, list = FALSE)
trainSet <- Model_data [inTrainFloor,]
test_valid<- Model_data[-inTrainFloor,]

set.seed(123)
smp_siz = floor(0.5*nrow(test_valid))
test_ind = sample(seq_len(nrow(test_valid)), size = smp_siz) 

testSet = test_valid[test_ind,] 
validSet =test_valid[-test_ind,]  


################################################################################## SVM ####
set.seed(123)
system.time(svm_long <- svm(LONGITUDE ~ . -BUILDINGID - FLOOR - LOCATION - LATITUDE, data = trainSet))

#Output
summary(svm_long)

#Predict the svm model
svm_long_pred  <- predict(svm_long, newdata=testSet)
svm_long_predV <- predict(svm_long, newdata=validSet)

#Metrics
#Predictions compared to test data
print(metrics_long_svm<-postResample(svm_long_pred, testSet$LONGITUDE))
print(metrics_long_svm2 <-mape(testSet$LONGITUDE, svm_long_pred))

#Save
saveRDS(svm_long,"C:/Users/46768/Documents/Dota/Wi-Fi position/SVM_Longitude_Model.rds")

########################################################################## Ranger (Random Forest)
set.seed(123)
system.time(ranger_long <- ranger(LONGITUDE ~ .-BUILDINGID - FLOOR - LOCATION - LATITUDE, trainSet, importance = "permutation", case.weights = T))

#Output
summary(ranger_long)

#Test the ranger model
ranger_long_pred <- predict(ranger_long, testSet)
ranger_long_predV <- predict(ranger_long, validSet)


#Metrics
#Predictions compared to test data
print(metrics_long_ranger<-postResample(ranger_long_pred$predictions, testSet$LONGITUDE)) 
metrics_long_ranger2<-mape(testSet$LONGITUDE, ranger_long_pred$predictions)

# Predictions compared to valid data
print(metrics_long_ranger<-postResample(ranger_long_predV$predictions, validSet$LONGITUDE))
print(metrics_long_rangerV2 <-mape(validSet$LONGITUDE, ranger_long_predV$predictions))

#Save Ranger Longitude model
saveRDS(ranger_long,"C:/Users/46768/Documents/Dota/Wi-Fi position/Ranger_Longitude_Model.rds")

###################################################################################### KNN ####
set.seed(123)
system.time(knn_long <-knnreg(LONGITUDE~.,data = trainSet[, -which(names(trainSet) %in% c("LATITUDE","BUILDINGID",
                                                                                          "FLOOR", "LOCATION"))]))
#Test model
knn_long_pred  <- predict(knn_long, testSet)
knn_long_predV <- predict(knn_long, validSet)

#Metrics
print(postResample(knn_long_pred, testSet$LONGITUDE))
print(postResample(knn_long_predV, validSet$LONGITUDE))



######################################################################## Errors #####################
#Data frame to plot & compare longitude & latitude with real location
Lat_Long_Diff <- data.frame(
  LONG.RANGER <- ranger_long_pred$predictions,
  LONG.KNN <-knn_long_pred,
  LONG.SVM <- svm_long_pred,
  LAT.SVM <- svm_lat_pred,
  LAT.RANGER<- ranger_lat_pred$predictions,
  LAT.KNN <- knn_lat_pred,
  LATITUDE <- testSet$LATITUDE,
  LONGITUDE <- testSet$LONGITUDE,
  BUILDING = testSet$BUILDINGID,
  FLOOR = testSet$FLOOR)
  
####################################################################### LATITUDE ERRORS ####
#Only latitude
  plot_ly(Lat_Long_Diff, x = ~LONGITUDE, y = ~LATITUDE, type = "scatter", name = "Real location") %>%
    add_trace(Lat_Long_Diff, y = ~LAT.SVM, name = "SVM prediction")%>%
    add_trace(Lat_Long_Diff,y = ~LAT.RANGER,name = "Ranger prediction") %>%
    add_trace(Lat_Long_Diff, x = ~LONG.KNN, name = "KNN prediction") %>%
    layout(title = "Predicted latitude")

  
#DF to plot latitude errors
Diff.lat.knn<- testSet$LATITUDE - knn_lat_pred
Diff.lat.Ranger <- testSet$LATITUDE - ranger_lat_pred$predictions 
Diff.lat.svm <- testSet$LATITUDE - svm_lat_pred
  
Z <- data.frame(v1=Diff.lat.knn, v2=Diff.lat.Ranger, v3=Diff.lat.svm)
colnames(Z) <- c("KNN","RANGER", "SVM")
Z<-melt(Z)
  
#Errors latitude 
  ggplot(Z,aes(x=value,fill=variable))  +
    labs(fill="") +
    geom_density(alpha=0.2) +
    ggtitle("Error distribution latitude") +
    theme_void()
  
  
###################################################################### LONGITUDE ERRORS ####  
#Only longitude
  plot_ly(Lat_Long_Diff, x = ~LONGITUDE, y = ~LATITUDE, type = "scatter", name = "Real location") %>%
    add_trace(Lat_Long_Diff, x = ~LONG.SVM, name = "SVM prediction") %>%
    add_trace(Lat_Long_Diff, x = ~LONG.KNN, name = "KNN prediction") %>%
    add_trace(Lat_Long_Diff,x = ~LONG.RANGER,name = "Ranger prediction")%>%
    layout(title = "Predicted longitude")
  
#DF to plot longitude errors
Diff.long.knn    <- testSet$LONGITUDE - knn_long_pred
Diff.long.Ranger <- testSet$LONGITUDE - ranger_long_pred$predictions 
Diff.long.svm    <- testSet$LONGITUDE - svm_long_pred 

mean(Diff.long.knn)
mean(Diff.long.Ranger)
mean(Diff.long.svm )
mean(abs(Diff.long.knn))

G <- data.frame(v1=Diff.long.knn, v2=Diff.long.Ranger, v3=Diff.long.svm )
colnames(G) <- c("KNN","RANGER","SVM")
G<-melt(G)

H <- data.frame(v1=Diff.long.Ranger, v2=Diff.long.svm)
colnames(H) <- c("RANGER","SVM")
H <-melt(H)

# Plot errors longitude
  ggplot(H,aes(x=value,fill=variable))  +
    labs(fill="") +
    geom_density(alpha=0.2) +
    ggtitle("Error distribution longitude") +
    theme_void()
  


  
