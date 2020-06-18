install.packages("tree")
install.packages("party")
install.packages("rpart")
install.packages("car")
install.packages("mlbench")
install.packages("mboost")
install.packages("textir")
install.packages("class")
install.packages("e1071")
install.packages("randomForest")
install.packages("h2o")

library(car)

### Q1 ###

# In my dataset i had chosen Block has Class Attribute becasuse after seeing my dataset i came to conclusion that this is all about virus in the chicago
# SO if we had known at what place the virus is more we can make some preventive measures in that areas to save peple from diseases.
# So if there are some mosquitoes killers sprays like hit which are not available in O hare blocks they can use this data class attributes and implement
# their products in that areas and gets profits and also save health of people.

### Mean, Median, Mode & sd ###

getwd()
sumitha <- read.csv("bank-additional-full.csv")
View(sumitha)


sumi <- sumitha[1:20000,c(1,2,3,4,6,7,15,16)]

str(sumi)
sumi <- na.omit(sumi)

### In order to findout mean, median, mode we need to have attributes in numeric format.

summary(sumi)

summary(sumi$age, na.rm = True)
range(sumi$age)
quantile(sumi$age)
mean(sumi$age)
median(sumi$age)

#mode(sumi$age)
getmode <- function(v)
{
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
result <- getmode(sumi$age)
print(result)

range(sumi$emp.var.rate)
quantile(bank$emp.var.rate)
mean(bank$emp.var.rate)
median(bank$emp.var.rate)

summary(bank$emp.var.rate)
#mode(virus$emp.var.rate)
mode<-(bank$emp.var.rate)
temp<-table(as.vector((mode)))
names(temp)[temp==max(temp)]
#View(mode)
sd(bank$emp.var.rate)


# sd for emp.var.rate is low

# Here block is the class atribute
# it is in the Charatcer so converting into numeric

#virus$BLOCK<-as.numeric(virus$BLOCK)

### Scatter plot ###

# 

plot (sumi$loan,sumi$job, main = " Scatterplot" , xlab = "loan", ylab = "job", pch = 20, col=" blue")

plot (sumi$loan,sumi$education, main = " Scatterplot" , xlab = "loan", ylab = "job", pch = 20, col=" red")


# from plots we can say that there is no max correlation between attributes.


### KNN ###

install.packages("car")
install.packages("mlbench")
install.packages("mboost")
install.packages("textir")
install.packages("class")
install.packages("e1071")
##################################################
#       k Nearest Neighbor       #
##################################################


library(textir) ## needed to standardize the data
library(MASS)   ## a library of example datasets

getwd()
mitha1 <- read.csv("bank-additional-full.csv")
 
mitha2 <- mitha1[1:20000,c(1,2,3,4,6,7,15,16)]

mitha <- na.omit(mitha2)

str(mitha)


par(mfrow=c(1,3), mai=c(.3,.6,.1,.1))
plot( age ~ loan , data=mitha, col=c(grey(.2),2:6))
plot(education ~ loan, data=mitha, col=c(grey(.2),2:6))
plot(marital ~ loan, data=mitha, col=c(grey(.2),2:6))

n=length(mitha$loan)
nt=500
set.seed(123) ## to make the calculations reproducible in repeated runs
train <- sample(1:n,nt)


## x <- normalize(virusq4[,c(4,3)])

x=mitha[,c(8,1)]
x[,1]=(x[,1]-mean(x[,1]))/sd(x[,1])
x[,2]=(x[,2]-mean(x[,2]))/sd(x[,2])

x[1:3,]

### KNN Algorithm ###

library(class)  
nearest3 <- knn(train=x[train,],test=x[-train,],cl=mitha$loan[train],k=1)
nearest5 <- knn(train=x[train,],test=x[-train,],cl=mitha$loan[train],k=5)
data.frame(mitha$loan[-train],nearest3,nearest5)

## ploting them to see how these works

par(mfrow=c(1,2))

## plot for k=3 (single) nearest neighbor

plot(x[train,],col=mitha$loan[train],cex=.8,main="3-nearest neighbor")
points(x[-train,],bg=nearest3,pch=21,col=grey(.9),cex=1.25)

## plot for k=5 nearest neighbors

plot(x[train,],col=mitha$loan[train],cex=.5,main="5-nearest neighbors")
points(x[-train,],bg=nearest5,pch=21,col=grey(.9),cex=1.25)

## calculating the proportion of correct classifications on this one
## training set

pcorrn1=100*sum(mitha$loan[-train]==nearest3)/(n-nt)
pcorrn5=100*sum(mitha$loan[-train]==nearest5)/(n-nt)
pcorrn1
pcorrn5

## cross-validation (leave one out)
pcorr=dim(10)
for (k in 1:10) {
  pred=knn.cv(x,mitha$loan,k)
  pcorr[k]=100*sum(mitha$loan==pred)/n
}

pcorr


######################################
########## NaviBayes ##############
######################################

library(mlbench)


## barplots for specific issue

plot(as.factor(mitha[,4]))

title(main=" bank marketing ", xlab="loan", ylab="#reps")## looking around that dataset

mitha[,"train"] <- ifelse(runif(nrow(mitha))<0.75,1,0)

mitha$train = as.factor(mitha$train)

str(mitha)

## Getting col number of train / test indicator column (needed later)

trainColNum <- grep('train', names(mitha))

## separating training and test sets and removing training column before we model the data

trainmitha <- mitha[mitha$train==1,-trainColNum]
testmitha <- mitha[mitha$train==0,-trainColNum]
testmitha

str(mitha)

## building the Naive Bayes model

## Loading e1071 library and invoking naiveBayes method

library(e1071)
gsr <- naiveBayes(loan ~ .,data = trainmitha)
gsr
summary(gsr)
str(gsr)

gsr_test_predict <-predict(gsr, testmitha[,-14])

## Building confusion matrix

table(pred=gsr_test_predict,true=testmitha$loan)


# Naive Bayes Classification produces the good results when compared with KNN Algorithm.
# While comparing the KNN Algorithm is the lazy Model and predicts the values of target variable with the help of Euclidean distance and works only for smaller datasets with less attributes. While predicting values in real time environment this type of models fails lot of times to give accurate results. As the k value increases the accuracy decreases.
# Whereas Naive Bayes classification is an eager learner model and predicts the values more accurately in the real time environment which produce accurate probabilities and works best for larger datasets.
# For KNN Algorithm while K=3 and K=5 we got results as
# pcorrn1
# [1] 50
# > pcorrn5
# [1] 42.85714
# This clearly states that if k value increases accuracy decreases.
# For Naïve Bayes model the results are more accurate when compared with KNN because output clearly says that most of the predictions are right like how many times particular team had won when compared with train and test sets.


### Random Forest ###


# In random forest we need to have or convert the classs attribute as numeric but the other variabblesshould be factors.

getwd()
mitha1 <- read.csv("bank-additional-full.csv")
mitha2 <- mitha1[1:5000,c(1,2,3,4,6,7,15,16)]
mitha2
mitha <- na.omit(mitha2)
str(mitha)

# mitha[sapply(mitha, is.numeric)] <- lapply(mitha[sapply(mitha, is.numeric)], as.factor)

# mitha <- as.numeric(mitha$loan)

ind <- sample(2, nrow(mitha), replace=TRUE, prob=c(0.75, 0.25))
trainmitha  <- mitha[ind==1,]
testmitha  <- mitha[ind==2,]


## Here I am predicting the RESULT variable with all other variables in the data with
## randomForest() Function.

library(randomForest)

rf <- randomForest(loan~ ., data=trainmitha, ntree=100, proximity=TRUE)
table(predict(rf), trainmitha$loan)
print(rf)
attributes(rf)

## Ploting the RandomForest

plot(rf)

## I am obtaining the importance of variables with functions importance() and varImpPlot()

importance(rf)
varImpPlot(rf)

## Finally, the built random forest is tested on test data, and the result is checked with functions table().

mithaPred <- predict(rf, newdata=testmitha)
table(mithaPred, testmitha$loan)

# Random Forest algorithm produces the good results because it is the combination of tree algorithms whereas others are only single tree algorithms.
# The random forest prediction algorithm is good because if we compare the outputs of simple tree and ctree algorithm along with Random Forest we can clearly say which is good model,
# especially in these cases of tree and ctree we are giving only few variables as inputs and getting the results which are appropriate and
# good also but in the case of random forest we are providing the larger no of inputs and getting the results similar to tree and ctree.
# So, getting accurate output by taking more inputs is the best model. If we see the outputs of tree, ctree and random forests.
# We are able to predict team 7 winnings with some probabilities in tree like 0.667 and 0.1 etc., by providing 4 inputs.
# In ctree we are able to predict team 7 winnings as 0/18 where we had given 4 inputs. In Random Forest we are able to predict team 7 winnings as 11/18 where we had given more than 10 inputs.
# By these results we came to conclusion that random forest is best model.

# Random Forest algorithm works better than the KNN and Naïve Bayes Predictions. KNN, Naïve Bayes, tree and Random Forest are used to predict the class attribute outputs by setting the train and test subsets
# and KNN is used only if the dataset is numeric, Naïve Bayes is used if dataset is categorical whereas Tree and Random forest works on both numeric or categorical or combination of both.
# In KNN we may delete some variables to perform the operation accurately but in Random forest no need of deleting any attribute and also handles the large amount of data with ease.
# Random Forest has 2 special features where other prediction algorithms doesn’t have,
# they are: Random Forest estimates the variable which is important while classification and also estimates the missing data when there is large missing data in dataset.
# Coming to results in previous lab while performing the naïve Bayes model prediction we got the team 7 winnings as 12/14 but there,
# we need to convert the data to categorical and in Random Forest we got team 7 winnings as 11/18 where the data set is combination of both numeric and categorical.
# So, we can say we have few benefits like no need to convert the dataset datatypes and can get good results by using Random Forest.



### Neural Networks ###


install.packages("neuralnet")
library(neuralnet)
library(ggplot2)
library(nnet)
library(dplyr)
library(reshape2)


set.seed(1234)


testmitha[sapply(testmitha, is.factor)] <- lapply(testmitha[sapply(testmitha, is.factor)], as.numeric)


# Converting observation class and LOAN into one vector.

labels <- class.ind(as.factor(testmitha$loan))

# Generic function to standardize a column of data.

standardizer <- function(x){(x-min(x))/(max(x)-min(x))}

# Performing Normalization predictors. We need lapply to do this.

testmitha[, 1:6] <- lapply(testmitha[, 1:6], standardizer)
    

# Reviewing the data for Normalization

# Combining labels and standardized predictors.

pre_process_mitha <- cbind(testmitha[,1:6], labels)

View(pre_process_mitha)

# Formula for the neuralnet using the as.formula function

fr <- as.formula(" loan ~ job+education")

# Creating a neural network object using the tanh function and two hidden layers of size 10 and 8.

mitha_net <- neuralnet(fr, data = pre_process_mitha, hidden = c(10,8), act.fct = "tanh", linear.output = FALSE)

# Ploting the neural network.

plot(mitha_net)

# I am using the compute function and the neural network object's net.result attribute for
# Calculating the overall accuracy of the  neural network.

mitha_preds <-  neuralnet::compute(mitha_net, pre_process_mitha[, 1:6])
origi_values <- max.col(pre_process_mitha[, 6:9])
pr.nn_2 <- max.col(mitha_preds$net.result)
print(paste("Model Accuracy: ", round(mean(pr.nn_mitha==origi_values)*100, 2), "%.", sep = ""))

### Accuracy 16.67% ###


# Neural Network Analysis is used to set certain no of weights to produce the correct classification for larger no of rows in the training set and predict the output accuracy for test set.
# If the accuracy you are getting is lesser, we need to change the no of iterations so that we can obtain more accuracy.
# Neural Network is good for more tolerance and noisy data but it takes more time for execution.
# Similarly, while we are performing the neural network task with our match dataset, we performed the task on both 67 % of train and 34% of test dataset.
# On the train dataset we got accuracy as 87.88% and we are able to predict the test dataset with 12 attributes as inputs and we got accuracy more than half like 51.52%.

# We had tuned our neural network to increase the quality of our model and reached almost the quality of Random Forest Model.
# We can say that we had reached our goal by coming closer to quality of random forest because while we had performed the Random Forest, we are able to predict the team 7 Winnings as 11/18 which is almost 60% accuracy.
# Our Neural network accuracy on test dataset is 51.52% which is almost nearer to 60% accuracy of Random Forest.
# We had obtained this not only by adjusting the hidden layers values but also by taking only numeric attributes as inputs and limiting the data.
# For the first time we had executed the program by taking the hidden layers (16,12) but we could not get the output and besides we got an error.
# So, we had continuously worked on this by changing different ranges of hidden layers like (10,8), (10,6) but we did not get the output and later we tried with (10,8,5) we are able to get the neural net but we did not get the output during model accuracy.
# But finally, we got output for (12,10) and accuracy as 51.52% with these we are also able to reach Random Forest accuracy.



### KMEANS ###

install.packages("cluster")  
install.packages("fpc")


testmitha[sapply(testmitha, is.factor)] <- lapply(testmitha[sapply(testmitha, is.factor)], as.numeric)

testmitha

testmitha$poutcome<-NULL

moon<-testmitha

str(moon)

moon$loan<- NULL

### At first, we are removing Block from the data to cluster.
### Now, we are appling function kmeans() to virus5, and storing the clustering result in match.kmeans.result.
### Here we are taking 3 Clusters in the code below.

## virus5[apply(sapply(virus5, is.finite), 1, all),]

match.kmeans.result <- kmeans(moon, 3)


### The clustering result is then compared with the class label (BLOCk) to check whether similar
### objects are grouped together.  The  result shows that cluster with Blocks are not having  any small degree overlapped with each other.
### whereas few blocks can be separated from all others clusters.

table(testmitha$loan, match.kmeans.result$cluster)

### We are ploting the clusters and their centers. we found the results of k-means clustering may vary
### from run to run, due to random selection of initial cluster centers.

par(mfrow=c(1,2), mai=c(.3,.6,.1,.1))

plot(moon[c("job", "education")], col = match.kmeans.result$cluster)

points(match.kmeans.result$centers[,c("job", "education")], col = 1:3, pch = 8, cex=2)


### Hierarchical Clustering ###

### We are performing hierarchical clustering with hclust().
### We are first drawing a sample of 100 records from the match data, so that the clustering plot will
### not be overcrowded. Same as before, variable Block is removed from our data.
### After that,we are applying hierarchical clustering to the data.

idx.match <- sample(1:dim(testmitha)[1], 100)

sun <- testmitha[idx.match,]
sun$loan <- NULL
hc.match <- hclust(dist(sun), method="complete")

plot(hc.match, hang = -1, labels=testmitha$loan[idx.match])

### we are cutting the tree into 3 clusters

rect.hclust(hc.match, k=3)
groups.match <- cutree(hc.match, k=3)

## Similar to the above clustering of k-means, the cluster "Tie" and "NO Result" can be easily separated from
## the other two clusters, and clusters "by runs" and "by wickets" overlap.

# Kmeans Algorithm: It is a partitioning algorithm. It is used to form the group of clusters for respective data based on the similarities
# and dissimilarities in the data and in this model the cluster is represented by centres, we need to pre define the no of clusters manually
# in the code and in order to execute this operation without any errors we ned to consider the variables of same scale and
# we need to Perform Pre-Processing: Normalizing the data and Post-Processing.
# The Problems in Kmeans clustering are: it can’t handle noise and can’t accommodate shapes for clusters.

# Hierarchical Clustering Algorithm: Which forms nested clusters in the form of tree. The tree records the sequence of data.
# The strength is not to specify any no of clusters and performs clustering based on distance matrix.
# The Problems in this technique are once decision is made that can’t be undone to combine clusters. This can’t handle noise and Large amount of data.

# DBscan Algorithm: It is a density-based algorithm used to form clusters in arbitrary shapes based on the given data
# by specifying density and the minimum points which we specify manually in the code. DBscan is good in handling noise and obtains shapes for clusters.
# DBscan is one of the good approaches for clustering because here DBscan can form no of clusters according to the levels of our class attribute.
# DBscan is well for small quantity of data whereas not well for high amount of data, we may not get appropriate output by varying densities and only takes inputs as numeric data.

# 2. Ans: Results depend upon the algorithm which you had chosen. We think K-Means best than DBscan and Hierarchical for our data.
# We think K-Means is best analysis for our model because our model determined the no of clusters which we had specified without any overlapping
# and we can easily differentiate the 3 clusters. The Kmeans result shows that cluster by runs, by wickets, are having small degree overlapped with each other.
# whereas Tie and No Result can be separated from all others clusters. The Similar output we got for the DBScan but it failed to determine the no of clusters according to the class attribute.
# The Hierarchical Model also failed to form clusters according to the given data. So, we think the Kmeans is good for our data.

# 3. Ans: According to our dataset we had taken Win_Type as the class attribute because the goal of match is to know who had won the match and with how many runs or wickets they had won.
# So, we had taken Win_Type as class attribute and in that we have total 4 categories by runs, by wickets, No Result, Tie.
# The maximum no of clusters can be formed is 4 and minimum is 1. We can successfully graph and tabled 3 clusters with the help of Kmeans.  
# On the other hand, DBScan failed to determine max 4 no of clusters from the given data and overlapping can be easily seen in the graph and tabular column.
# Among the three algorithms, Kmeans algorithm gives some significant clustering of our dataset. Because our dataset is combination of factors and numeric data, we can successfully perform the Kmeans on this data but we need to consider only the numeric variable for DBScan.
# So, we got best output for Kmeans and we think this as good Model for our Dataset.

# 4. Ans: The Results from Kmeans show that the data can be classified by three clusters without overlapping in the graph and we can easily recognize the three clusters from the graph.
# With the help of above analysis, the Cricket organization will be able to determine the No result cluster and go to those matches details and checks weather conditions of those areas
# where the match has held and might change the place of match in the coming future to a good climatic conditions place and where majority of people are cricket lovers in order to held the match properly.
# So, the management can earn some revenues and also can attain people’s attention towards cricket.
# From the above analysis the management could make decisions on the team and player performances and if they are lacking any skills like handling pressure during the tied matches the management could provide them with those counselling sessions.
# In this way the management can make Business, political and medical decisions on the results.




###  Density-based Clustering ###

library(fpc)

ds.moon <- dbscan(moon, eps=0.42, MinPts=3)

### we are choosing Win_Type has the Class variable.
### Comparing the clusters with the original class labels


table(ds.moon$cluster, testmitha$loan)

plot(ds.moon, moon)

### Displaying the clusters in a scatter plot using the first and 4 th column of the data.


plot(ds.moon, moon[c(1,4)])

### plotcluster plotting.


plotcluster(moon, ds.moon$cluster)

### The clustering model can be used to label new data,
### Based on the similarity between new data and the clusters. The following example draws a sample
### of 10 objects from tp data and adds small noises to them to make a new dataset for labeling.
### The random noises are generated with a uniform distribution using function runif().

### Create a new dataset for labeling

set.seed(435)
idx.win <- sample(1:nrow(testmitha), 10)
newData <- testmitha[idx.win,-1]
newData <- newData + matrix(runif(10*4, min=0, max=0.2), nrow=10, ncol=4)

### Labeling new data

myPred.moon <- predict(ds.moon, moon, newData)

### Plot the result with new data as asterisks

plot(moon[c(1,4)], col=1+ds.moon$cluster)
points(newData[c(1,4)], pch="*", col=1+myPred, cex=3)

### Check cluster labels

table(myPred.moon, testmitha$loan[idx.win])

### C TREE ###

getwd()
virus4 <- read.csv("./West_Nile_Virus__WNV__Mosquito_Test_Results.csv")
virus4 <- na.omit(virus4)
# virus4$RESULT <- as.factor(virus4$RESULT)
virus4$BLOCK <- as.numeric(virus4$BLOCK)

str(virus4)
virus4

set.seed(1234)
v4 <- sample(2, nrow(virus4), replace=TRUE, prob=c(0.25, 0.75))
trainvirus <- virus4[v4==1,]
testvirus <- virus4[v4==2,]

## I am building a decision tree, and checking the prediction results.
## Here v4a specifies that RESULT is the target variable and all other variables are independent variables.


library(party)
v4a <-  BLOCK~ WEEK + SEASON.YEAR
v4ctree <- ctree(v4a, data=trainvirus)

## Now I am making some predictions.
## I am doing this by comparing the train and test data outputs.
## Iam also performing the confusion matrix.
## which is a table where we can get the true and predicted values in matrix form.

table(predict(v4ctree), trainvirus$RESULT)


# Non -Technical manager decisions

# From the results of question 2 and 3 we came to conclusion that Naïve Bayes Model is the better model that fits for our dataset for predicting the probabilities of target predictor.
# Generally, for a non-technical manager he just needs whether we are getting the results or not and how much percentage we are successful in getting results that is enough
# for non-technical manager and he does not require on what factors we are getting results because they don’t expect whether the work which we did is perfect or not.
# They see only our commitment and we can able to overcome obstacles or not and they can give you promotions. In this way a non-technical manager can make decisions that help for both employee and organization.


# business and political decisions

# The results of the Naïve Bayes Model provides the manager to take some most useful decisions that might be related to business,
# medical fields such as if we consider business field manager can come to point that out of total 13 teams in IPL the team which has more wins is the best team and might own that IPL team for the next season
# so this statistics helps manager to buy a team. And if we consider the medical field the team which had more wins had played more games and played well so that team players might have more injuries and may require some rest to that team players and
# provide some health check-ups. While coming to political decisions the team which has more wins that team players are having good skills and the managers may neglect the players who played well in other teams and
# select the players only from winning team for country team and show some favouritism for the most winning team and this leads politics in cricket. In this way manager can make decisions.

# COrrelation

# By Going through the given Match dataset, we had noticed that there might me a relationship between City_Name and Host_Country
# because the City where the cricket match is being held entirely depends on the Country Hosted. So, for this sake we had chosen these 2 attributes.

# From our Indian premiere league dataset, we had considered Match_Winner_Id as the class attribute because ultimately at the end of cricket match,
# we are enthusiastic to know that which team had won the match. So, for this purpose we had chosen Match_Winner_Id as the Class attribute.


#  Three analytical questions that management could ask are:
#  1. Which player has won highest man of the match awards.
#  2. Which team has the highest winning percentage.
#  3. Which team has highest wining chances either first batting team or first fielding team

# why we are eliminating rows theory.
# There is an IS_Result variable that duplicate the content of Win_Type because in the cricket win type represents with
# how many runs or wickets had the team won. So, this declares whether the match has result or not.
# So, we are eliminating the IS_Result variable and reducing the data.
# In this dataset We can perform binning the data for the variable IS_Superover by this we can know how many super over matches had occurred.

# Theory for y selecting numeric attributes.
# We had selected these variables because the content of this variables is in numeric and may create problem in future
# because in cricket the match is held between two teams and finally one team will be the winner
# and here in our dataset these 3 variables are in numeric which represents the team id’s
# and it would be difficult to know the name of the team because we need to go through another dataset to identify the name of team and
# It is time taking process. So, we are converting this numeric datatype to character datatype.















