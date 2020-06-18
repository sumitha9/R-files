
getwd()
sampleofp <- read.csv("OFP.csv")[1:200,c(2,7,8,9,11,15,17,18,20)]
summary(sampleofp)
#Let us see if there are any na alues
is.na(sampleofp)
sample1 <- na.omit(sampleofp)
nrow(sample1)
#NOw we dont have any na values
str(sample1)
cor()
#1  Which class attribute does your dataset have?  For what business or political
#purposes would someone be interested in this class attribute?
#In our dataset the class attribute is Result, our dataset has data on  mosquitos test so the class attribute 
#give the result whether the result is postive our negative

#List the Minimum, Maximum, Mean, Median, Mode, Q1, Q3, and Standard Deviation for two 
#appropriate attributes.  
#Which attribute has the smaller standard deviation?
##To find the minimum maximum mean mode and standard deviation we should find correlation between the attributes
##Let's check correlation between the attributes 
sample1$hlth = as.numeric(sample1$hlth) 
sample1$medicaid = as.numeric(sample1$medicaid)
sample1$privins= as.numeric(sample1$privins)
###Correlation between two attributes ###

cor(sample1$ofp,sample1$medicaid,use="all.obs",method=c("pearson"))
cor(sample1$adldiff,sample1$medicaid,use="all.obs",method=c("pearson"))
summary(sample1$medicaid)
summary(sample1$ofp)
#For each of the attributes above, make a scatterplot and describe in detail how
#each of the attributes is correlated to the identified class attribute.
# Let's plot this!
plot(sample1$hlth,sample1$medicaid, main="Scatterplot ofp vs. medicod",xlab="medicod",ylab="ofp")
plot(sample1$hlth,sample1$adldiff, main="Scatterplot ofp vs. medicod",xlab="medicod",ylab="ofp")
#Separate the dataset into 25% training data and 75% test data.  
#Remove the class attribute values from the test data set.  Then prepare the 
#training data to run with two of the classifiers below.  
#Neural network
#Naive Bayes
#k Nearest Neighbor
#Random Forest
###########################################################################
#                      k Nearest Neighbor                                 #
###########################################################################
install.packages("car")
install.packages("mlbench")
install.packages("mboost")
install.packages("textir")
install.packages("class")
install.packages("e1071")

library(textir) ## needed to standardize the data
library(MASS)  
sample1

ind <- sample(2, nrow(sample1), replace=TRUE, prob=c(0.75, 0.25))
tdata <- sample1[ind==1,]
testData <- sample1[ind==2,]
par(mfrow=c(3,3), mai=c(.3,.6,.1,.1))
plot(hlth ~ medicaid, data=sample1, col=c(grey(.2),2:6))
plot( adldiff ~ medicaid, data=sample1, col=c(grey(.2),2:6))
n=length(sample1$medicaid)
n
nt=180
set.seed(1) ## to make the calculations reproducible in repeated runs
train <- sample(1:n,nt)##dont change sample here its formula
###Normaize the dataset

x=sample1[,c(2,8)]
x[,1]=(x[,1]-mean(x[,1]))/sd(x[,1])
x[,2]=(x[,2]-mean(x[,2]))/sd(x[,2])

x[1:3,]

library(class)  
nearest1 <- knn(train=x[train,],test=x[-train,],cl=sample1$medicaid[train],k=1)
nearest5 <- knn(train=x[train,],test=x[-train,],cl=sample1$medicaid[train],k=5)
data.frame(sample1[-train],nearest1,nearest5)

## plot them to see how it worked
par(mfrow=c(1,2))
## plot for k=1 (single) nearest neighbor
plot(x[train,],col=sample1$medicaid[train],cex=.8,main="1-nearest neighbor")
points(x[-train,],bg=nearest1,pch=21,col=grey(.9),cex=1.25)
## plot for k=5 nearest neighbors
plot(x[train,],col=sample1$medicaid[train],cex=.8,main="5-nearest neighbors")
points(x[-train,],bg=nearest5,pch=21,col=grey(.9),cex=1.25)
legend("topright",legend=levels(sample1$medicaid),fill=1:6,bty="n",cex=.75)
## calculate the proportion of correct classifications on this one 
## training set

pcorrn1=100*sum(sample1$medicaid[-train]==nearest1)/(n-nt)
pcorrn5=100*sum(sample1$medicaid[-train]==nearest5)/(n-nt)
pcorrn1
pcorrn5
## cross-validation (leave one out)

pcorr=dim(10)
for (k in 1:10) {
  pred=knn.cv(x,sample1$medicaid,k)
  pcorr[k]=100*sum(sample1$medicaid==pred)/n
}
pcorr
###########################################################################
#     NAIVE BAYES (conditional probability)                               #
###########################################################################
## load mlbench library
library(mlbench)

## load dataset
sample1

## Dividing data into Test set and training set

sample1[,"train"] <- ifelse(runif(nrow(sample1))<0.75,1,0)

str(sample1)

sample1$train = as.factor(sample1$train)

sample1$medicaid = as.factor(sample1$medicaid)

traincolNum <- grep('train', names(sample1)) ##trying to find column with name train

## Separating trainign set and test set

trainData <- sample1[sample1$train==1,-traincolNum]
ncol(trainData)
testData <- sample1[sample1$train==0, -traincolNum]

## Invoking naiveBayes method.

library(e1071)

output <- naiveBayes(medicaid~.,data = sample1)

output

str(output)

summary(testData)

nb_test_predict <-predict(output, testData[,-10])

##F Building confusion matrix

table(pred=nb_test_predict,true=testData$medicaid)
################################################################
###################### Random Forest ##############################
install.packages("randomForest")
library(randomForest)
sampleofp <- read.csv("OFP.csv")
sampler <- sampleofp[1:200,c(2,9,11,17,18,20)]
sampler
##sample1[sapply(sample1, is.numeric)] <- lapply(examsam[sapply(examsam, is.numeric)], as.factor)

sampler$medicaid <- as.numeric(sampler$medicaid)
sam <- sampler
str(sam)

#NOw we dont have any na values

ind <- sample(2, nrow(sam), replace=TRUE, prob=c(0.75, 0.25))
traindata <- sam[ind==1,]
testdata <- sam[ind==2,]

rf <- randomForest(medicaid ~ ., data=traindata, ntree=50, proximity=TRUE)

table(predict(rf), traindata$medicaid)
print(rf)
attributes(rf)

## After that, we plot the error rates with various number of trees.

plot(rf)

## The importance of variables can be obtained with functions importance() and varImpPlot()

importance(rf)
varImpPlot(rf)

## Finally, the built random forest is tested on test data, and the result is checked with functions
## table() and margin(). The margin of a data point is as the proportion of votes for the correct
## class minus maximum proportion of votes for other classes. Generally speaking, positive margin
## means correct classification.

Pred <- predict(rf, newdata=testdata)
table(Pred, testdata$medicaid)
plot(margin(rf, testdata$medicaid))
################################################################################################
################################neural network#################################
install.packages("neuralnet")
install.packages("ggplot2")
install.packages("nnet")
install.packages("dplyr")
install.packages("reshape2")
library(neuralnet)
library(ggplot2)
library(nnet)
library(dplyr)
library(reshape2)

data <- read.csv("OFP.csv")
data1 <- data[1:100, c(1,2,9,10,18)]
data1
str(data1)
## data1$medicaid <- as.numeric(data1$medicaid) ### should not be done.. In neural network class attribute shouldn't be converted to numeric
cor(data1)  ## max correlation between adldiff and ofp, ofp and medicaid, medicaid and adldiff
set.seed(123)
# Converting RESULT into one vector.
labels <- class.ind(as.factor(data1$medicaid))
standardizer <- function(x){(x-min(x))/(max(x)-min(x))}
data1[, 1:4] <- lapply(data1[, 1:4], standardizer)
data1  # Review the data and see what the standardization function has done
# Normalizing / standardizing the predictors
pre_process_data <- cbind(data1[,c(1:4)], labels)
pre_process_data
# Define a formula for neuralnet
f <- as.formula(" no + yes ~ ofp + adldiff")
sumi_net <- neuralnet(f, data = pre_process_data, hidden = c(4, 2), act.fct = "tanh", linear.output = FALSE)
# Plotting the neural network and testing it with Test Dataset.
plot(sumi_net)
## Finding the accuracy of the model built.
sumi_preds <-  neuralnet::compute(sumi_net, pre_process_data[,1:4 ])
pre_process_data
origi_vals <- max.col(pre_process_data[,5:6])
pr.nn_2 <- max.col(sumi_preds$net.result)
print(paste("Model Accuracy: ", round(mean(pr.nn_2==origi_vals)*100, 2), "%.", sep = ""))
## The accuracy of this model is "58%"
## [1] "Model Accuracy: 58%."
## Which states the predictions are not that accurate, but is better than the Random Forest Algorithm.

#Using the compute function and the neural network object's net.result attribute, 
# let's calculate the overall accuracy of the  neural network.

data_preds <-  neuralnet::compute(data_net, pre_process[, 1:5])
origi_vals <- max.col(pre_process[, 5])
pr.nn_2 <- max.col(data_preds$net.result)
print(paste("Model Accuracy: ", round(mean(pr.nn_2==origi_vals)*100, 2), "%.", sep = ""))

#########  Using the test data set you have just generated, 
#remove  the class attribute and run two of the clustering algorithms below.  
#Produce a biplot to identify attribute-to-attribute clusters
#################################KMEANS################################
## We will be using these libraries

install.packages("h2o")  ## Should already be installed
install.packages("cluster")  ## Should be installed by default
install.packages("fpc")  ## For density-based clustering


## At first, we remove CLASS from the data to cluster. After that, we apply function kmeans() to
##DATASET and store the clustering result in kmeans.result. The cluster number is set to 3 in the
## code below.

mitha1 <- read.csv("OFP.csv")
#mitha[sapply(mitha, is.factor)] <- lapply(mitha[sapply(mitha, is.factor)], as.numeric)
mitha2 <- mitha1[1:100,c(14,2,9,10,18)]
is.na(mitha2)
sum(is.na(mitha2))
mitha <- na.omit(mitha2)
sum(is.na(mitha))
mitha$medicaid <- NULL
str(mitha)
kmeans.result <- kmeans(mitha, 4)
str(mitha)
print(kmeans.result)
kmeans.result$cluster

## The clustering result is then compared with the class label (Species) to check whether similar
## objects are grouped together.  The  result shows that cluster \setosa" can be easily separated from the other clusters, and
## that clusters "versicolor" and "virginica" are to a small degree overlapped with each other.

table(mitha2$medicaid, kmeans.result$cluster)

## Let's plot the clusters and their centers. Note that there are four dimensions in the data 
## and that only the first two dimensions are used to draw the plot below.  Some black points 
## close to the green center (asterisk) are actually closer to the black center in the
## four dimensional space. We also need to be aware that the results of k-means clustering may vary
## from run to run, due to random selection of initial cluster centers.

plot(mitha[c("age", "ofp")], col = kmeans.result$cluster)
points(kmeans.result$centers[,c("age", "ofp")], col = 1:3, pch = 8, cex=2)
###########################################################################################
########################### Hierarchical Clustering ##########################

## We will perform hierarchical clustering with hclust()
## We first draw a sample of 40 records from the iris data, so that the clustering plot will 
## not be overcrowded. Same as before, variable Species is removed from the data. After that, 
## we apply hierarchical clustering to the data.
Data1 <- read.csv("OFP.csv")[1:500,c(14,2,9,10,18)]

Data <- na.omit(Data1)
idx <- sample(1:dim(Data)[1], 15)
DataSample <- Data[idx,]
DataSample$medicaid <- NULL
hc <- hclust(dist(DataSample), method="complete")

plot(hc, hang = -1, labels=Data1$medicaid[idx])

## Let's cut the tree into 3 clusters
rect.hclust(hc, k=3)
groups <- cutree(hc, k=3)

## Similar to the above clustering of k-means, the cluster "setosa" can be easily separated from 
## the other two clusters, and clusters "versicolor" and "virginica" overlap slightly.

######################################################################
####################################DBScan ##########################################


## The DBSCAN algorithm provides a density based clustering for numeric data. 
## This means it groups objects into one cluster if they are connected to one another by a densely 
## populated area. There are two key parameters in DBSCAN :
## 1. eps: reachability distance, which denes the size of neighborhood; and
## 2. MinPts: minimum number of points.
## If the number of points in the neighborhood of point a is no less than MinPts, then a is a 
## dense point. All the points in its neighborhood are density-reachable from a and are put into 
## the same cluster as a.  The strengths of density-based clustering are that it can discover 
## clusters with various shapes and sizes and is insensitive to noise. In contrast, k-means usually finds
## spherical clusters with similar sizes.

install.packages("fpc")

library(fpc)
Data1 <- read.csv("OFP.csv")[1:500,c(2,9,10,18)]
Data2 <- na.omit(Data1)
Data2
Data3 <- Data2[-4]   ## Remove class
str(Data3)
View(Data3)
ds <- dbscan(Data3, eps=0.98, MinPts=15)

## Let's compare the clusters with the original class labels
table(ds$cluster, Data2$medicaid)
unique(Data2$medicaid)

## In the table, "1" to "3" in the first column are three identified clusters, while "0" stands 
## for noises or outliers, i.e., objects that are not assigned to any clusters. The noises are 
## shown as black circles in the plot below:
plot(ds, Data3)

## Or let's display the clusters in a scatter plot using the first and fourth columns of the data.
plot(ds, Data2[c(1,4)])

## Even better, let's use plotcluster.  Yeah!
plotcluster(Data3, ds$cluster)

## Now let's get crazy and do some predicting!  The clustering model can be used to label new data, 
## based on the similarity between new data and the clusters. The following example draws a sample 
## of 10 objects from iris and adds small noises to them to make a new dataset for labeling. 
## The random noises are generated with a uniform distribution using function runif().

## Create a new dataset for labeling

set.seed(123)
idx <- sample(1:nrow(Data2), 10)
newData <- Data2[idx,-4]
newData <- newData + matrix(runif(10*4, min=0, max=0.2), nrow=10, ncol=4)

## Label new data

myPred <- predict(ds, Data3, newData)

## Plot the result with new data as asterisks

plot(Data2[c(1,3)], col=1+ds$cluster)
points(newData[c(1,3)], pch="*", col=1+myPred, cex=3)

## Check cluster labels

table(myPred,Data2$medicaid[idx])


