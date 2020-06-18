getwd()
student1 <- read.csv("studentsheet.csv")
#reading the dataset
nrow(student1)
ncol(student1)
##here I reduced my dataset to minimum attributes
#student2[sapply(student2, is.factor)] <- lapply(student2[sapply(student2, is.factor)], as.numeric)

cor(student1)
View(student1)

student <- na.omit(student1)
student
is.na(student)
##lets see co-orrelation between the attributes
cor(student)
##from the correlation results there is strong relationship between failures and G3,
##and G3 , studytime. I am choosing these two attributes to find the mim max mode and median
summary(student)

##question1##

##question2

plot(student$guardian, student$failures, main="Scatterplot failures and G3",xlab="Failures",ylab="G3")
plot(student$guardian, student$studytime, main="studytime and failures",xlab="failures",ylab="studytime")

##from the graph we can say that 

###question3#####
###########################################################################
#                      k Nearest Neighbor                                 #
###########################################################################
install.packages("car")
install.packages("mlbench")
install.packages("mboost")
install.packages("textir")
install.packages("class")
install.packages("e1071")
library(textir)
library(MASS)
##spliting data into testing set and training set 
ind <- sample(2, nrow(student), replace=TRUE, prob=c(0.75, 0.25))
traindata <- student[ind==1,]
testData <- student[ind==2,]
par(mfrow=c(3,3), mai=c(.3,.6,.1,.1))
plot(failures ~ guardian, data=student, col=c(grey(.2),2:6))
plot(failures ~ studytime, data=student, col=c(grey(.2),2:6))
n=length(student$failures)
nt=300
set.seed(1) ## to make the calculations reproducible in repeated runs
train <- sample(1:n,nt)
## x <- normalize(fgl[,c(4,1)])
student[sapply(student, is.factor)] <- lapply(student2[sapply(student2, is.factor)], as.numeric)

x=student[,c(2,3)]
x[,1]=(x[,1]-mean(x[,1]))/sd(x[,1])
x[,2]=(x[,2]-mean(x[,2]))/sd(x[,2])

x[1:3,]

library(class)  
nearest1 <- knn(train=x[train,],test=x[-train,],cl=student$failures[train],k=1)
nearest5 <- knn(train=x[train,],test=x[-train,],cl=student$failures[train],k=5)
data.frame(student$failures[-train],nearest1,nearest5)

## plot them to see how it worked
par(mfrow=c(1,2))
## plot for k=1 (single) nearest neighbor
plot(x[train,],col=student$failures[train],cex=.8,main="1-nearest neighbor")
points(x[-train,],bg=nearest1,pch=21,col=grey(.9),cex=1.25)
## plot for k=5 nearest neighbors
plot(x[train,],col=student$failures[train],cex=.8,main="5-nearest neighbors")
points(x[-train,],bg=nearest5,pch=21,col=grey(.9),cex=1.25)
#legend("topright",legend=levels(student$failures),fill=1:6,bty="n",cex=.75)
## calculate the proportion of correct classifications on this one 
## training set

pcorrn1=100*sum(student$failures[-train]==nearest1)/(n-nt)
pcorrn5=100*sum(student$failures[-train]==nearest5)/(n-nt)
pcorrn1
pcorrn5
## cross-validation (leave one out)

pcorr=dim(10)
for (k in 1:10) {
  pred=knn.cv(x,student$guardian,k)
  pcorr[k]=100*sum(student$guardian==pred)/n
}
pcorr
###########################################################################
#     NAIVE BAYES (conditional probability)                               #
###########################################################################
## load mlbench library
library(mlbench)

## load dataset
sky1 <- student
View(student)
sky <- na.omit(sky1)
## Dividing data into Test set and training set
nrow(sky)
ncol(sky)
sky[,"train"] <- ifelse(runif(nrow(sky)<0.75,1,0))
train <- na.omit(train)
str(sky)
View(sky)
sky$train = as.factor(sky$train)

sky$failures = as.factor(sky$failures)
traincolNum <- grep('train', names(sky)) ##trying to find column with name train
traincolNum
## Separating trainign set and test set

trainData <- sky[sky$train==1,-traincolNum]
ncol(trainData)
testData <- sky[sky$train==0, -traincolNum]



## Invoking naiveBayes method.

library(e1071)

output <- naiveBayes(failures ~.,data = sky)

output

str(output)

summary(testData)

nb_test_predict <-predict(output, testData[,-7])

##F Building confusion matrix

table(pred=nb_test_predict,true=testData$failures)
################################################################
###################### Random Forest ##############################
install.packages("randomForest")
library(randomForest)
student2 <- read.csv("studentsheet.csv")
star1 <- student2

star <- na.omit(star1)


#star[sapply(star, is.numeric)] <- lapply(star[sapply(star, is.numeric)], as.factor)

star$failures<-as.numeric(star$failures)
star <- na.omit(star1)
str(star)

#NOw we dont have any na values
x=star[,c(2,4)]

ind <- sample(2, nrow(star), replace=TRUE, prob=c(0.75, 0.25))
traindatarf <- star[ind==1,]
testdatarf <- star[ind==2,]

rf <- randomForest(failures ~ ., data=traindatarf, ntree=10, proximity=TRUE)
rf
View(traindatarf)

table(predict(rf), traindatarf$failures)
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

Pred <- predict(rf, newdata=testdatarf)
table(Pred, testdatarf$failures)
summary(testdatarf$failures)
plot(margin(rf, testdatarf$failures))
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
set.seed(234)
star1 <- student
ncol(star1)
#Let us see if there are any na alues

 match <- na.omit(star1)
 nrow(match)
 ncol(match)
 #Let us see if there are any na alues
 
 #NOw we dont have any na values
 str(match)
 sum(is.na(match))
 #newsample[sapply(newsample, is.factor)] <- lapply(newsample[sapply(newsample, is.factor)], as.numeric)
 ##convert observation and result into one vector
 labels <- class.ind(as.factor(match$failures))
 # Write a generic function to standardize a column of data.
 labels
 standardizer <- function(x){(x-min(x))/(max(x)-min(x))}
 # Now standardize the predictors. We need lapply to do this.
 str(match)
 match[, 1:6] <- lapply(match[, 1:6], standardizer)
 match       # Review the data and see what the standardization function has done
 pre_processmatch <- cbind(match[,1:6], labels)
 View(pre_processmatch)
 ncol(match)
 nrow(match)
 # Define the formula for the neuralnet using the as.formula function
 
 f <- as.formula("0 + 1 ~ studytime + guardain ")
 
 data_match <- neuralnet(f, data = pre_processmatch, hidden = c(4,3), act.fct = "tanh", linear.output = FALSE)
 
 # Let's plot the neural network.
 
 plot(data_match)
 data_preds <-  neuralnet::compute(data_match, pre_processmatch[, 1:6])
 origi_vals <- max.col(pre_processmatch[, 5])
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
 
 ###################### k Means ######################
 
 ## At first, we remove CLASS from the data to cluster. After that, we apply function kmeans() to
 ##DATASET and store the clustering result in kmeans.result. The cluster number is set to 3 in the
 ## code below.
 
 student2
 Data <- na.omit(student2)
 Data$failures <- NULL
 kmeans.result <- kmeans(Data, 2)
 print(kmeans.result)
 kmeans.result$cluster
 
 ## The clustering result is then compared with the class label (Species) to check whether similar
 ## objects are grouped together.  The  result shows that cluster \setosa" can be easily separated from the other clusters, and
 ## that clusters "versicolor" and "virginica" are to a small degree overlapped with each other.
 
 table(student2$failures, kmeans.result$cluster)
 
 ## Let's plot the clusters and their centers. Note that there are four dimensions in the data 
 ## and that only the first two dimensions are used to draw the plot below.  Some black points 
 ## close to the green center (asterisk) are actually closer to the black center in the
 ## four dimensional space. We also need to be aware that the results of k-means clustering may vary
 ## from run to run, due to random selection of initial cluster centers.
 
 plot(Data[c("G3", "absences")], col = kmeans.result$cluster)
 points(kmeans.result$centers[,c("G3", "absences")], col = 1:3, pch = 8, cex=2)
 ###########################################################################################
 ########################### Hierarchical Clustering ##########################
 
 ## We will perform hierarchical clustering with hclust()
 ## We first draw a sample of 40 records from the iris data, so that the clustering plot will 
 ## not be overcrowded. Same as before, variable Species is removed from the data. After that, 
 ## we apply hierarchical clustering to the data.
 Data1 <- student2
 Data <- na.omit(Data1)
 idx <- sample(1:dim(Data)[1], 5)
 DataSample <- Data[idx,]
 DataSample$failures <- NULL
 hc <- hclust(dist(DataSample), method="complete")
 
 plot(hc, hang = -1, labels=Data1$failures[idx])
 
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
 
 library(fpc)
 arun1 <- student2

 arun3 <- na.omit(arun1)
 arun3$failures = as.numeric(arun3$failures)
 str(arun3)
 arun <- arun3[-5] ## Remove class
 str(arun)
 ds <- dbscan(arun, eps=0.42, MinPts=13)
 ## Let's compare the clusters with the original class labels
 table(ds$cluster, arun3$failures)
 ## In the table, "1" to "3" in the first column are three identified clusters, while "0" stands 
 ## for noises or outliers, i.e., objects that are not assigned to any clusters. The noises are 
 ## shown as black circles in the plot below:
 plot(ds, arun3)
 
 ## Or let's display the clusters in a scatter plot using the first and fourth columns of the data.
 plot(ds, arun3[c(1,4)])
 
 ## Even better, let's use plotcluster.  Yeah!
 plotcluster(arun, ds$cluster)
 
 ## Now let's get crazy and do some predicting!  The clustering model can be used to label new data, 
 ## based on the similarity between new data and the clusters. The following example draws a sample 
 ## of 10 objects from iris and adds small noises to them to make a new dataset for labeling. 
 ## The random noises are generated with a uniform distribution using function runif().
 
 ## Create a new dataset for labeling
 
 set.seed(135)
 idx <- sample(1:nrow(arun3), 10)
 newData <- arun3[idx,-5]
 newData <- newData + matrix(runif(10*4, min=0, max=0.2), nrow=10, ncol=4)
 
 ## Label new data
 
 myPred <- predict(ds, arun3, newData)
 
 ## Plot the result with new data as asterisks
 
 plot(arun3[c(1,4)], col=1+ds$cluster)
 points(newData[c(1,4)], pch="*", col=1+myPred, cex=3)
 
 ## Check cluster labels
 
 table(myPred,arun3$RESULT[idx])
 
 
 
 
 
