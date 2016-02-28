# Prediction of Excercise Quality
A. C. Dennerley  
February 25, 2016  



The data set has been generously provided by it's researchers.  More information on the data is available <a href="http://groupware.les.inf.puc-rio.br/har">here</a>.  

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. <a href="http://groupware.les.inf.puc-rio.br/work.jsf?p1=11201">Qualitative Activity Recognition of Weight Lifting Exercises</a>. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. 

###Data Processing

Loading data files


```r
dat.build <- if(file.exists('train.csv')) {
  fread('train.csv', na.strings=c("NA","","#DIV/0!"))
} else {
  fread('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv',
        na.strings=c("NA","","#DIV/0!"))
}
```

Data set is split into training and testing sets.


```r
buildIndex <- createDataPartition(dat.build$classe, p=0.8, list=FALSE)
dat.train <- dat.build[ buildIndex, ]
dat.test  <- dat.build[-buildIndex, ]
```

Variables with no relevance to prediction are removed.  Removed variables include row indicies, time/date of observation (but not specific time within a given excercise).


```r
dat.train[,`:=`( V1=NULL, user_name=NULL, new_window=NULL, num_window=NULL,
                 raw_timestamp_part_1=NULL, cvtd_timestamp=NULL )]
dat.test[ ,`:=`( V1=NULL, user_name=NULL, new_window=NULL, num_window=NULL,
                 raw_timestamp_part_1=NULL, cvtd_timestamp=NULL )]
```

Any character columns that can be converted to numeric are converted.  Columns with large quantities of 'NA' values increase the variance of a fitted model without neccessarily improving it's accuracy.  At the very least, columns with all 'NA' values are removed.  Removing columns with at least 50% 'NA' values reduces the risk of overfitting without sacrificing accuracy.

Variables that provide summary statistics on the whole window of physical activity are a special case.  These variables are not readily available in a potential test set, thus are excluded to reduce risk of overfitting.


```r
datSize <- dim(dat.train)
NAMES <- names(dat.train)

thresholdNANs    <- 0
thresholdNARatio <- 0.5

for(NAME in NAMES){
  # Recover numeric columns that were read as character
  if(class(dat.train[[NAME]]) == "character") {
    if(sum(!grepl('^(-)?([0-9]*)(\\.)?([0-9]+)$',dat.train[[NAME]]),na.rm=TRUE) == thresholdNANs) {
      dat.train[[NAME]] <- as.numeric( dat.train[[NAME]] )
      dat.test[[NAME]]  <- as.numeric( dat.test[[NAME]]  )
    }
  }
  # Remove any columns with above threshold NA values
  naRatio <- sum(is.na(dat.train[[NAME]])) / datSize[1]
  if(naRatio > thresholdNARatio) {
    dat.train[[NAME]] <- NULL
    dat.test[[NAME]]  <- NULL
  }
}
```

Removing variables with near-zero variance further reduces overfitting without much negative consequence.


```r
nzv.cols <- nearZeroVar(dat.train)
dat.train[,(nzv.cols) := NULL]
dat.test[,(nzv.cols) := NULL]
```

Confounding variables are identified by their mutually high correlations.


```r
NAMES <- names(dat.train)
datSize <- dim(dat.train)

thresholdRR <- 0.95

# Copy numeric/integer data.
dat.num <- dat.train[,sapply(NAMES,function(NAME){
  (class(dat.train[[NAME]])=='numeric' | class(dat.train[[NAME]])=='integer')
}),with=FALSE]
# Build matrices of the correlations between variables
dat.cor <- cor(dat.num)
# Print correlated pairs.
names.cor <- names(dat.num)
for(i in 1:dim(dat.cor)[1]){
  for(j in 1:i){
    if(dat.cor[i,j]^2 > thresholdRR & i!=j) {
      print(paste(dat.cor[i,j]^2," for ",names.cor[i]," and ",names.cor[j]))
    }
  }
}
```

```
## [1] "0.962459545968892  for  total_accel_belt  and  roll_belt"
## [1] "0.984201635962184  for  accel_belt_z  and  roll_belt"
## [1] "0.951077849852308  for  accel_belt_z  and  total_accel_belt"
## [1] "0.96580483278299  for  gyros_dumbbell_z  and  gyros_dumbbell_x"
```

###Modelling

While generalized linear models or especially Fourier transformed models be used effectively, the reponse variable is a factor with 6 levels.  Testing shows 54 variables as predictors of 6 possible outcomes.  This is a an ideal case for trees, using conditional structures to ignore the large quantities of junk data.  To clarify each of the 6 outcomes should be identifiable by a subset range of values in a subset of predictors, and which predictors are most relevant may differ by outcome.  For A,B,C,D,E, 5 predictors may narrow possible outcomes to A,C, and D at which point, predictors relevant to distinguish C for A for example are not useful.  Rather than regress how much to discount the conditionally uneccessary predictors, it's more effective to consider the predictors conditionally with trees.  

Within classification by trees, there are a number of existing <a href="http://topepo.github.io/caret/modelList.html">R-packages</a> that provide effective algorithms to use.  Boosting with trees is a natural choice, taking many weak predictors to create strong predictors but may consequently amplify noise.  Considering proper technique for an excercise via accelerometers makes noisy data an expectation.  Instead a random forest algorithm is used.  Though a random forest method is generally time consuming, the accuracy will be substantially higher than that of a single tree.  The 'e1071', 'randomForest' and 'foreach' packages can be used to apply random forest such that the trees are built in parallel; giving a very accurate algorithm for the computation time (faster than boosting in series).

A brief note about cross-validation:  80% of the data is in the training set.  The 8-fold cross validation here uses 70% of the data (maintaining most complete subsets of data per user) and repeats 8 times leaving out (and testing on) a different 10% fold (1/8 of training set) each time.  The last 20% is held back for testing.

For contrast, a classification & regression tree model is performed first.


```r
set.seed(98086)
# Use a classification tree to predict variables
modelRpart <- train( classe ~ ., data=dat.train, method='rpart',
                     trControl=trainControl(method = "cv", number = 8) )
modelRpart
```

```
## CART 
## 
## 15699 samples
##    53 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (8 fold) 
## Summary of sample sizes: 13736, 13737, 13736, 13738, 13737, 13736, ... 
## Resampling results across tuning parameters:
## 
##   cp          Accuracy   Kappa       Accuracy SD  Kappa SD  
##   0.03515799  0.5037908  0.35152221  0.009468181  0.01276356
##   0.05889334  0.4428867  0.25376458  0.063388511  0.10690858
##   0.11535381  0.3236438  0.05985475  0.041956065  0.06399614
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.03515799.
```

```r
predictTest <- predict(modelRpart, newdata=dat.test)
confusionMatrix(predictTest,reference=dat.test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1025  322  280  284   97
##          B   20  254   25  119  114
##          C   67  183  379  240  185
##          D    0    0    0    0    0
##          E    4    0    0    0  325
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5055          
##                  95% CI : (0.4897, 0.5212)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3543          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9185  0.33465  0.55409   0.0000  0.45076
## Specificity            0.6498  0.91214  0.79160   1.0000  0.99875
## Pos Pred Value         0.5105  0.47744  0.35958      NaN  0.98784
## Neg Pred Value         0.9525  0.85108  0.89369   0.8361  0.88982
## Prevalence             0.2845  0.19347  0.17436   0.1639  0.18379
## Detection Rate         0.2613  0.06475  0.09661   0.0000  0.08284
## Detection Prevalence   0.5119  0.13561  0.26867   0.0000  0.08386
## Balanced Accuracy      0.7841  0.62339  0.67285   0.5000  0.72476
```

Now the random forest.


```r
set.seed(98086)
# register cores/threads compute trees in parallel for the forest
registerDoParallel()
modelRF <- train( classe ~ ., data=dat.train, method='parRF',
                  trControl=trainControl(method = "cv", number = 8) )
modelRF
```

```
## Parallel Random Forest 
## 
## 15699 samples
##    53 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (8 fold) 
## Summary of sample sizes: 13736, 13737, 13736, 13738, 13737, 13736, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9935665  0.9918616  0.002194282  0.002775976
##   27    0.9936300  0.9919417  0.002358851  0.002984635
##   53    0.9867509  0.9832389  0.003549753  0.004491897
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 27.
```

Applying the model to the test data.


```r
predictTest <- predict(modelRF, newdata=dat.test)
confusionMatrix(predictTest,reference=dat.test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    3    0    0    0
##          B    0  754    3    0    0
##          C    0    1  679    4    0
##          D    0    1    2  639    0
##          E    0    0    0    0  721
## 
## Overall Statistics
##                                         
##                Accuracy : 0.9964        
##                  95% CI : (0.994, 0.998)
##     No Information Rate : 0.2845        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.9955        
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9934   0.9927   0.9938   1.0000
## Specificity            0.9989   0.9991   0.9985   0.9991   1.0000
## Pos Pred Value         0.9973   0.9960   0.9927   0.9953   1.0000
## Neg Pred Value         1.0000   0.9984   0.9985   0.9988   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1922   0.1731   0.1629   0.1838
## Detection Prevalence   0.2852   0.1930   0.1744   0.1637   0.1838
## Balanced Accuracy      0.9995   0.9962   0.9956   0.9964   1.0000
```

This final model effectively predicts the quality of activity.


```r
# Save subsequent model
saveRDS(modelRF,"QOA_model.rds")
```

###Package & System Information

R version 3.2.3    data.table 1.9.6    knitr_1.12.3

caret 6.0-64    randomForest 4.6-12    rpart 4.1-10

doParallel 1.0.10    e1071 1.6-7    foreach 1.4.3
