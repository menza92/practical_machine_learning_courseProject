# Prediction of Excercise Quality
A. C. Dennerley  
February 25, 2016  



In summary, four sets of accelerometers/gyroscopes are attached to a subject's dumbell, forearm, arm and belt, as the subject lifts free weights (arm curls).  Data was collected on subjects performing the exercise properly under the supervision of a trainer (labelled "A"), and collected on the subjects performing the excercise with a highly common mistake in technique (labelled "B","C","D","E").  Here, statistical learning is applied to develope a model which can predict the quality of technique in the exercise.

The data set has been generously provided by it's researchers.  More information on the data is available <a href="http://groupware.les.inf.puc-rio.br/har">here</a>.  

A much more detailed explanation of the background, data gathering procedures and apparatus is detailed in the paper below.

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
## [1] "0.962827672949323  for  total_accel_belt  and  roll_belt"
## [1] "0.984013055139922  for  accel_belt_z  and  roll_belt"
## [1] "0.951091569173788  for  accel_belt_z  and  total_accel_belt"
## [1] "0.966317350096198  for  gyros_dumbbell_z  and  gyros_dumbbell_x"
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
##   cp         Accuracy   Kappa       Accuracy SD  Kappa SD  
##   0.0352470  0.5049356  0.35332090  0.009175604  0.01241307
##   0.0599911  0.4609978  0.28368225  0.063223968  0.10481583
##   0.1153538  0.3229433  0.05883229  0.041226368  0.06292743
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.035247.
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
##    2    0.9936940  0.9920228  0.001609147  0.002035972
##   27    0.9940760  0.9925059  0.001742833  0.002205528
##   53    0.9864955  0.9829132  0.003958806  0.005012830
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
##          A 1116    7    0    0    0
##          B    0  751    3    0    0
##          C    0    1  678    5    5
##          D    0    0    3  638    4
##          E    0    0    0    0  712
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9929          
##                  95% CI : (0.9897, 0.9953)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.991           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9895   0.9912   0.9922   0.9875
## Specificity            0.9975   0.9991   0.9966   0.9979   1.0000
## Pos Pred Value         0.9938   0.9960   0.9840   0.9891   1.0000
## Neg Pred Value         1.0000   0.9975   0.9981   0.9985   0.9972
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1914   0.1728   0.1626   0.1815
## Detection Prevalence   0.2863   0.1922   0.1756   0.1644   0.1815
## Balanced Accuracy      0.9988   0.9943   0.9939   0.9950   0.9938
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
