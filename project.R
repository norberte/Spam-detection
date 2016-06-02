# Data importing and manipulation -----------------------------------------
spam <- read.csv("spam_data.csv", header=FALSE) # importing the data set
# manually adding the column names
colnames(spam) <- c("word_freq_make","word_freq_address","word_freq_all","word_freq_3d","word_freq_our","word_freq_over","word_freq_remove","word_freq_internet","word_freq_order","word_freq_mail","word_freq_receive","word_freq_will","word_freq_people","word_freq_report","word_freq_addresses","word_freq_free","word_freq_business","word_freq_email","word_freq_you","word_freq_credit","word_freq_your","word_freq_font","word_freq_000","word_freq_money","word_freq_hp","word_freq_hpl","word_freq_george","word_freq_650","word_freq_lab","word_freq_labs","word_freq_telnet","word_freq_857","word_freq_data","word_freq_415","word_freq_85","word_freq_technology","word_freq_1999","word_freq_parts","word_freq_pm","word_freq_direct","word_freq_cs","word_freq_meeting","word_freq_original","word_freq_project","word_freq_re","word_freq_edu","word_freq_table","word_freq_conference","char_freq_semicolon","char_freq_roundBracket","char_freq_squareBracket","char_freq_exclamationMark","char_freq_dollarSign","char_freq_hashtag","capital_run_length_average","capital_run_length_longest","capital_run_length_total", "decision")

spamDecisions <- spam[,58] # decision if an e-mail is spam or not
spamData <- spam[,-58] # all the data without the response variable

set.seed(1779) # just a random feed for reproducible results
ind <- sample(1:4601, 1000) # create a sample of 1000 numbers

train <- spam[-ind,] # 3601 observations for training
test <- spam[ind,] # 1000 observations for testing

trainDecision <- spamDecisions[-ind] # training response variable only
testDecision <- spamDecisions[ind] # testing response variable only
# Hierarchical clustering -------------------------------------------------
scale_Spam <-  scale(spamData) # scaling the data set
distmat <- dist(scale_Spam) # creating the distance matrix

# creating 3 different hierarchical clusters with complete, single and average linkages
completeLink <- hclust(distmat, method="complete") 
singleLink <- hclust(distmat, method="single")
averageLink <- hclust(distmat, method="average")

# plots of the 3 clusters
plot(completeLink)
plot(averageLink)
plot(singleLink)

# cutting the trees at k=2, since there should be 2 groups
sres <- cutree(singleLink, k=2)
ares <- cutree(averageLink, k=2)
cres <- cutree(completeLink, k=2)

# mis-classification table for the 3 kinds of hierarchical clusters 
table(sres, spamDecisions)
table(ares, spamDecisions)
table(cres, spamDecisions)

# Single linkage misclassification rate = 0.3938274
# Average linkage misclassification rate = 0.3936101
# Complete linkage misclassification rate = 0.3936101

# Variable selection for LDA -------------------------------------------
library(MASS)

attach(train)
# create a linear model with the training data set
simlm <- lm(decision ~., data=train)
summary(simlm)

attach(spam)
# create a linear model with the entire data set
simlmFull <- lm(decision~., data = spam)
summary(simlmFull)

# use the stepAIC funcion to perform backward variable selection on the training and entire data set's linear models
step <- stepAIC(simlm, direction = "backward")
fullStep <- stepAIC(simlmFull, direction = "backward")

# LDA with model from Variable selection for LDA -------------------------------------------
# create linear discriminant analysis model with the training linear model previously created
ldamod <- lda(step$call$formula, data = train)
ldaResult <- predict(ldamod, newdata = test[,-58]) # predict response values for the test set
table(ldaResult$class, testDecision) # mis-classification table

(29+81)/1000 # 0.11 misclassification rate (11.0 %)

# LDA cross-validation ----------------------------------------------------
library(MASS)
ldacv <- lda(test[,-58], testDecision, CV=TRUE) # lda model with cross validation
table(testDecision, ldacv$class)  # mis-classification table
(86+32)/1000
# misclassification rate = 0.118

# QDA with model from Variable selection before ---------------------------------------------------------------------
# create quadratic discriminant analysis model with the training linear model previously created
qdamod <- qda(fullStep$call$formula, data = spam) 
qdaResult <- predict(qdamod, newdata = spam) # predict response values for the test set
table(qdaResult$class, spamDecisions) # mis-classification table
(699+89)/4601
# misclassification rate = 0.1712671

# QDA cross-validation ----------------------------------------------------
library(MASS)
qdacv <- qda(spamData, spamDecisions, CV = TRUE) # qda model with cross validation
table(spamDecisions, qdacv$class) #  mis-classification table
(691+87)/4601
#misclassification rate =  0.1690937

# K-means clustering ------------------------------------------------------
install.packages("clue")
library("clue")
# perform k-means clustering with k=2 (2 response groups) on the training data
kspam <- kmeans(train, 2)
clustore <- matrix(0, nrow=nrow(train), ncol=ncol(train)) 
wsstore <- NULL
for(i in 1:5){
  dum <- kmeans(scale(train), i, nstart=25)
  clustore[, i] <- dum$cluster # create the cluster values
  wsstore[i] <- dum$tot.withinss # choose the values that are within
} 

# predict results from the k-means clustering
results <- cl_predict(kspam, newdata = test)
table(results, test$decision)
(349 + 15)/(nrow(test))
# misclassification rate = 0.364

# knn Classification ------------------------------------------------------
library(gclus)
library(MASS)
library(class)

# perform knn classification with k = 5
kresults <- knn(train[,-58],test[,-58],train[,58],k=5)
kresults
table(kresults,test[,58])
(90+96)/(nrow(test))
# misclassification rate = 0.186 (18.6 %)

# Logistic regression -----------------------------------------------------
# 1.
attach(train)
# perform logistic regression using all of the attributes in the model, and have binomial response value
logModel <- glm(decision ~.,family=binomial(link='logit'),data=train)
summary(logModel) # get the summary of the model
results <- predict(logModel,newdata=test[,-58],type='response')
testPredictions <- ifelse(results > 0.5,1,0) # get the right predictions
table(testPredictions, testDecision)
(38+43)/1000
# Logistic reg. formula: decision ~ . : Misclassification rate = 0.081 (8.1 %)

# 2.
# stepAIC model, which gets in another logistic regression model
#stepLog <- stepAIC(logModel, direction = "backward")
#improvedLogModel = glm(stepLog$call$formula, data = train)
#summary(improvedLogModel)
#results <- predict(improvedLogModel,newdata=test[,-58],type='response')
#testPredictions <- ifelse(results > 0.5,1,0)
#table(testPredictions, testDecision)
#(12+93)/1000
# Logistic reg. w/ stepAIC backward direction : Misclassification rate = 0.105 (10.5 %)

# 3.
# Use logistic regression with a backward selection linear model
improvedModel = glm(decision ~. - capital_run_length_average - `char_freq_hashtag` - `char_freq_squareBracket` - `char_freq_roundBracket` - word_freq_table - word_freq_original - word_freq_cs - word_freq_direct - word_freq_1999 - word_freq_415 - word_freq_857 - word_freq_telnet - word_freq_labs - word_freq_lab - word_freq_addresses - word_freq_report - word_freq_receive - word_freq_mail - word_freq_email - word_freq_people - word_freq_3d - word_freq_all - word_freq_make - word_freq_address - word_freq_hpl - word_freq_650 - word_freq_parts , family=binomial(link='logit'),data=train)
summary(improvedModel)
results <- predict(improvedModel,newdata=test[,-58],type='response')
testPredictions <- ifelse(results > 0.5,1,0)
table(testPredictions, testDecision)
(32+45)/1000
# Logistic reg. w/ improved model (threshold at Alpha = 0.05) : Misclassification rate = 0.077 (7.7 %)

# Artificial Neural Network -----------------------------------------------
library(neuralnet)
attach(train)

# the below line was created with help from this resource : http://stackoverflow.com/questions/29555473/creating-formula-using-very-long-strings-in-r
formula <- paste("decision~", paste(sprintf("`%s`", colnames(spam[-58])), collapse="+"))
# formula is just the model with all the predictor attributes concatination

# creating the neural network
nnSpam <- neuralnet(formula, data=train, linear.output = FALSE, hidden=40)
nnresults <- compute(nnSpam, test[,-58]) # compute the results (predictions) from the neural network model

# plot(nnSpam) # plot the neural network (might not work on every computer, since the plot is really large)
table(testDecision, nnresults$net.result>0.5)

# Ran the analysis with different hidden layers/ variable values
# Single-layer
# Misclassification rate = 0.077 (7.7 % w/  5 hidden variables, 1 hidden layer)
# Misclassification rate = 0.066 (6.6 % w/ 10 hidden variables, 1 hidden layer)
# Misclassification rate = 0.070 (7.0 % w/ 20 hidden variables, 1 hidden layer)
# Misclassification rate = 0.076 (7.6 % w/ 30 hidden variables, 1 hidden layer)
# Misclassification rate = 0.070 (7.0 % w/ 35 hidden variables, 1 hidden layer)
# Misclassification rate = 0.069 (6.9 % w/ 36 hidden variables, 1 hidden layer)
# Misclassification rate = 0.064 (6.4 % w/ 37 hidden variables, 1 hidden layer)
# Misclassification rate = 0.081 (8.1 % w/ 38 hidden variables, 1 hidden layer)
# Misclassification rate = 0.067 (6.7 % w/ 39 hidden variables, 1 hidden layer)
# Misclassification rate = 0.064 (6.4 % w/ 40 hidden variables, 1 hidden layer)
# Misclassification rate = 0.071 (7.1 % w/ 45 hidden variables, 1 hidden layer)
# Misclassification rate = 0.067 (6.7 % w/ 50 hidden variables, 1 hidden layer)

# Multi-layers
# 2 hidden layers with 5 and 5 hidden variables did not converge
# Misclassification rate = 0.068 (6.8 % w/ 2 hidden layers,  7 and  7 hidden variables)
# Misclassification rate = 0.073 (7.3 % w/ 2 hidden layers, 10 and 10 hidden variables)
# Misclassification rate = 0.074 (7.4 % w/ 2 hidden layers, 15 and 10 hidden variables)
# Misclassification rate = 0.067 (6.7 % w/ 2 hidden layers, 15 and 15 hidden variables)
# Misclassification rate = 0.078 (7.8 % w/ 2 hidden layers, 13 and 13 hidden variables)
# Misclassification rate = 0.074 (7.4 % w/ 2 hidden layers, 11 and 11 hidden variables)
# Misclassification rate = 0.067 (6.7 % w/ 2 hidden layers, 12 and 11 hidden variables)
# Misclassification rate = 0.089 (8.9 % w/ 2 hidden layers, 18 and 17 hidden variables)
# Misclassification rate = 0.087 (8.7 % w/ 2 hidden layers, 20 and 10 hidden variables)
# Misclassification rate = 0.076 (7.6 % w/ 2 hidden layers, 20 and 15 hidden variables)
# Misclassification rate = 0.075 (7.5 % w/ 2 hidden layers, 20 and 18 hidden variables)
# Misclassification rate = 0.069 (6.9 % w/ 2 hidden layers, 25 and 13 hidden variables)
# Misclassification rate = 0.076 (7.6 % w/ 2 hidden layers, 25 and 15 hidden variables)
# Misclassification rate = 0.083 (8.3 % w/ 2 hidden layers, 20 and 20 hidden variables)
# Misclassification rate = 0.085 (8.5 % w/ 2 hidden layers, 25 and 20 hidden variables)

# Misclassification rate = 0.082 (8.2 % w/ 3 hidden layers, 10, 10 and  5 hidden variables)
# Misclassification rate = 0.081 (8.1 % w/ 3 hidden layers, 10, 10 and 10 hidden variables)
# Misclassification rate = 0.085 (8.5 % w/ 3 hidden layers, 15, 10 and  5 hidden variables)
# Misclassification rate = 0.079 (7.9 % w/ 3 hidden layers, 15, 10 and 10 hidden variables)
# Misclassification rate = 0.077 (7.7 % w/ 3 hidden layers, 15, 15 and 10 hidden variables)
# Misclassification rate = 0.070 (7.0 % w/ 3 hidden layers, 15, 15 and 15 hidden variables)
# Misclassification rate = 0.090 (9.0 % w/ 3 hidden layers, 20, 10 and 10 hidden variables)
# Misclassification rate = 0.076 (7.6 % w/ 3 hidden layers, 20, 15 and 10 hidden variables)
# Misclassification rate = 0.099 (9.9 % w/ 3 hidden layers, 20, 15 and 15 hidden variables)

# Principle Component Analysis combined with classification trees -------------------------------------------
library(HSAUR2)
library(tree)

# Perform Principle component analysis on the entire spam data set
pcaSpam <- prcomp(spam[,-58], scale.=TRUE)
round(pcaSpam$rotation[,1], 2)

# data frame containing the rotation of the scores
dataFromPca <- data.frame(pcaSpam$x[,1])
trainResponse <- factor(spamDecisions)
# creating the classification tree with the scores obtained from PCA
pcaTree <- tree(trainResponse~., data = dataFromPca)
# summary of the classification tree
summary(pcaTree)
plot(pcaTree)
text(pcaTree)

# pruning the pca classification tree
set.seed(1779)
cvspam <- cv.tree(pcaTree, FUN=prune.misclass) # cross valadation
plot(cvspam, type="b")
p.cvspam <- prune.misclass(pcaTree, best=2) # choose to prune it at 2
plot(p.cvspam) # the pruned classification tree's splits
text(p.cvspam)
summary(p.cvspam) # summary of the pruned classification tree

# Misclassification rate = 0.149098 (14.9098 %) from the summary of the classification tree

# Trees ---------------------------------------------------------------
library(tree) # load in proper library
# put training data into data frame
trainingSet<- data.frame(train)
trainingSet$decision <- factor(trainingSet$decision)
#create classification tree using the training data set
tspam <- tree(trainingSet$decision~., data = trainingSet)
summary(tspam) # summary of the model
plot(tspam) # plot of the splits
text(tspam)

# Tree Cross validation & Pruning
set.seed(1779)
cvspam <- cv.tree(tspam, FUN=prune.misclass) # cross validation
plot(cvspam, type="b")
p.cvspam <- prune.misclass(tspam, best=9) # pruning
plot(p.cvspam) # plot of the pruned version's splits
text(p.cvspam) 
summary(p.cvspam)   # summary of the pruned model

# Tree prediction
# use test data to predict
treePredictions = predict(p.cvspam, newdata=test[,-58])
res <- NULL

# deciding on which group to classify to based on higher probability of belonging to that group
for(i in 1:nrow(treePredictions)){
  if(treePredictions[i,1] > treePredictions[i,2]){
    res[i] <- 0
  }else{
    res[i] <- 1
  }
}

table(res, testDecision)
(46+53)/1000
# Misclassification rate = 0.099 (9.9 %)

# Bagging -----------------------------------------------------------------
library(randomForest)
set.seed(1779)
attach(spam)
# build a bagging model using the entire data set with mtry being 57 because of 57 predictor variables
spambag <- randomForest(factor(decision)~., data=spamData, mtry=57, importance=TRUE)
spambag

(137+104)/4601

# Misclassification Rate = 0.05237992 (5.23%)

# Random forrest ----------------------------------------------------------
set.seed(1779)
# build a random forest model using the entire data set
trainRF <- randomForest(factor(decision)~., data=spamData, importance=TRUE)
trainRF
varImpPlot(trainRF)
importance(trainRF)

(141+78)/4601

# Misclassification Rate = 0.04759835 (4.75%)

# Regression --------------------------------------------------------------
######No Interactions########
library(MASS)
#Hand picked backward selection using P-values
simpleNoInterlm <- lm(decision ~ word_freq_make + word_freq_all + word_freq_3d + word_freq_our + word_freq_over + word_freq_remove + word_freq_internet + word_freq_order + word_freq_receive + word_freq_will + word_freq_free + word_freq_business + word_freq_email + word_freq_you + word_freq_credit + word_freq_your + word_freq_font + word_freq_000 + word_freq_money + word_freq_hp + word_freq_hpl + word_freq_george + word_freq_labs + word_freq_data + word_freq_85 + word_freq_1999 + word_freq_pm + word_freq_direct + word_freq_meeting + word_freq_original + word_freq_project + word_freq_re + word_freq_edu + word_freq_table + word_freq_conference + `char_freq_;` + `char_freq_(` + `char_freq_!` + `char_freq_$` + `char_freq_#` + capital_run_length_longest + capital_run_length_total, data=train)
summary(simpleNoInterlm)

#predicts using this model on the test set
simpleNoInterlmresults <- predict.lm(simpleNoInterlm, newdata = test)
#compares to the decision solumn the results of the regression
table(testDecision, round(simpleNoInterlmresults))
#calculates the misclassficaion rate
(4+82+29+6)/1000

#######12.1% misclassification

######Interactions#############
library(MASS)
#gets the model with all the interactions
simplelm <- lm(decision ~ word_freq_make + word_freq_all + word_freq_3d + word_freq_our + word_freq_over + word_freq_remove + word_freq_internet + word_freq_order + word_freq_receive + word_freq_will + word_freq_free + word_freq_business + word_freq_email + word_freq_you + word_freq_credit + word_freq_your + word_freq_font + word_freq_000 + word_freq_money + word_freq_hp + word_freq_hpl + word_freq_george + word_freq_labs + word_freq_data + word_freq_85 + word_freq_1999 + word_freq_pm + word_freq_direct + word_freq_meeting + word_freq_original + word_freq_project + word_freq_re + word_freq_edu + word_freq_table + word_freq_conference + `char_freq_;` + `char_freq_(` + `char_freq_!` + `char_freq_$` + `char_freq_#` + capital_run_length_longest + capital_run_length_total + .*., data=train)
summary(simplelm)

#uses stepAIC to select out the non signifiat interactions
#DO NOT RUN TAKES MANY HOURS
#You can load up the attached enviroment to get the data object
step = stepAIC(simplelm,direction = 'backward')
step$fitted.values
step$model

#predicts on the test set
simplelmresults <- predict.lm(step, newdata = test)
#comapres for to the actual decision
table(testDecision, round(simplelmresults))
#calculates the rate
(4+82+29+6)/1000
#######23% misclassification

#####Refined Interactions########
library(MASS)
#Using p-value backwad selection by hand on the less variables.
relm <- lm(decision ~ `char_freq_!` + word_freq_remove + `char_freq_$` + word_freq_hp + capital_run_length_average + 
             capital_run_length_average*`char_freq_$` + capital_run_length_average*word_freq_remove + capital_run_length_average*`char_freq_!` + 
             word_freq_hp*`char_freq_$` +
             word_freq_remove*`char_freq_$` +
             `char_freq_$`*`char_freq_!`, data=train)
summary(relm)

#predicts on the test data
relmresults <- predict.lm(relm,newdata = test)
#compares tot he decision
table(testDecision, round(relmresults))
#calculates the results
(1+4+202+10+2+1)/1000
########## 22.0% Misclassification##########

