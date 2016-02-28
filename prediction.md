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
## [1] "0.962826316239155  for  total_accel_belt  and  roll_belt"
## [1] "0.984098534719344  for  accel_belt_z  and  roll_belt"
## [1] "0.951337986076654  for  accel_belt_z  and  total_accel_belt"
## [1] "0.965878315458024  for  gyros_dumbbell_z  and  gyros_dumbbell_x"
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
##   0.03658211  0.5036626  0.35156157  0.01465303   0.02000041
##   0.06079217  0.4279789  0.22894762  0.06245878   0.10569359
##   0.11473075  0.3329459  0.07413321  0.04034080   0.06161477
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.03658211.
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
##    2    0.9923564  0.9903303  0.001981756  0.002507446
##   27    0.9919738  0.9898470  0.002259227  0.002857865
##   53    0.9841389  0.9799340  0.003320188  0.004204078
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
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
##          A 1115    1    0    0    0
##          B    1  755   12    0    0
##          C    0    3  672   12    0
##          D    0    0    0  629    0
##          E    0    0    0    2  721
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9921          
##                  95% CI : (0.9888, 0.9946)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.99            
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9947   0.9825   0.9782   1.0000
## Specificity            0.9996   0.9959   0.9954   1.0000   0.9994
## Pos Pred Value         0.9991   0.9831   0.9782   1.0000   0.9972
## Neg Pred Value         0.9996   0.9987   0.9963   0.9957   1.0000
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1925   0.1713   0.1603   0.1838
## Detection Prevalence   0.2845   0.1958   0.1751   0.1603   0.1843
## Balanced Accuracy      0.9994   0.9953   0.9889   0.9891   0.9997
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
