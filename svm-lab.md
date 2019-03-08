Untitled
================
Brooke Seibert
3/5/2019

1.  This question refers to Chapter 9 Problem 8 beginning on page 371 with the OJ data set which is part of the ISLR package.
    1.  Create a training sample that has roughly 80% of the observations. Use 'set.seed(19823)'.

``` r
set.seed(19823)
df <- tbl_df(OJ)
inTraining <- createDataPartition(df$Purchase, p = .8, list = F)
training <- df[inTraining, ]
testing  <- df[-inTraining, ]
```

    b. Use the `kernlab` package to fit a support vector classifier to the training data using `C = 0.01`. 

``` r
#Fit support vector classiﬁer to training data using cost=0.01, with Purchase as the response and the other variables as predictors. Use the summary() function to produce summary statistics, and describe the results obtained. 
fit_control <- trainControl(method = "repeatedcv",
                           number = 10, 
                           repeats = 3)
oj_train <- train(Purchase ~ .,
                     data = training,
                     method = "svmLinear",
                     trControl = fit_control,
                     tuneGrid = data.frame(C = 1:10))
oj_train
```

    ## Support Vector Machines with Linear Kernel 
    ## 
    ## 857 samples
    ##  17 predictor
    ##   2 classes: 'CH', 'MM' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (10 fold, repeated 3 times) 
    ## Summary of sample sizes: 771, 771, 772, 771, 772, 771, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   C   Accuracy   Kappa    
    ##    1  0.8272664  0.6341345
    ##    2  0.8303899  0.6403207
    ##    3  0.8323235  0.6442123
    ##    4  0.8303853  0.6398258
    ##    5  0.8307685  0.6407056
    ##    6  0.8292179  0.6372973
    ##    7  0.8307820  0.6409102
    ##    8  0.8303762  0.6397248
    ##    9  0.8311696  0.6414451
    ##   10  0.8319449  0.6430249
    ## 
    ## Accuracy was used to select the optimal model using the largest value.
    ## The final value used for the model was C = 3.

``` r
plot(oj_train)
```

![](svm-lab_files/figure-markdown_github/1b.%20support%20vector%20classifier%20SVC-1.png)

    c. Compute the confusion matrix for the training data. Report the overall error rates, sensitivity, and specificity. 

``` r
confusionMatrix(oj_train)
```

    ## Cross-Validated (10 fold, repeated 3 times) Confusion Matrix 
    ## 
    ## (entries are percentual average cell counts across resamples)
    ##  
    ##           Reference
    ## Prediction   CH   MM
    ##         CH 53.3  9.1
    ##         MM  7.7 29.9
    ##                             
    ##  Accuracy (average) : 0.8324

    d. Construct the ROC curve. 

``` r
#oj_lda <- lda(Purchase ~ ., data = training)
#fits <- predict(oj_lda)
#new_fits <- mutate(training, pprobs = predict(oj_lda)$posterior[, 2], default = if_else(Purchase == "Yes", 1, 0))
#p <- ggplot(data = new_fits, aes(d = default, m = pprobs))
#p + geom_roc(n.cuts = 0, col = "navy") + style_roc()

#heart_svc <- ksvm(ahd ~ ., data = training,
#                  type = "C-svc", kernel = 'vanilladot', prob.model = TRUE)
#fits_svc <- predict(heart_svc, newdata = training, type = "probabilities")
#svc_pred <- mutate(new_fits, svc_probs = fits_svc[, 2])
#p <- ggplot(data = svc_pred,
#            aes(d = default, m = pprobs))
#p + geom_roc(n.cuts = 0, col = "navy") +
#  geom_roc(aes(d = default, m = svc_probs), n.cuts = 0, col = "#6e0000") +
#  style_roc()
```

``` r
#oj_svm_d2 <- ksvm(Purchase ~ ., data = training,
#                     type = "C-svc", kernel = 'polydot', 
#                     kpar = list(degree = 2, scale = .1),
#                     C = 1, prob.model = T)
#fits_svm_d2 <- predict(oj_svm_d2, newdata = training, 
#                       type = "probabilities")
#svc_pred_d2 <- mutate(svc_pred, svc_probs_d2 = fits_svm_d2[, 2])
#p <- ggplot(data = svc_pred_d2,
#            aes(d = default, m = pprobs))
#p + geom_roc(n.cuts = 0, col = "#1b9e77") +
#  geom_roc(aes(d = default, m = svc_probs), n.cuts = 0, col = "#d95f02") +
#  geom_roc(aes(d = default, m = svc_probs_d2), n.cuts = 0, col = "#7570b3") +
#  style_roc()
```

    e. Use train function from caret package to find optimal cost  parameter (`C`) in range 0.01-10. Use `seq(0.01, 10, len = 20)`. 


    f. Compute the training and test classification error.


    g. Repeat (b) - (d) using an SVM with a polynomial kernel with degree 2.


    h. Which method would you choose?


    i. Repeat (b) - (d) using an SVM with a radial basis kernel. Train it. 
    j. Using the best models from LDA, SVC, SVM (poly), and SVM (radial), compute the test error. 


    k. Which method would you choose?

1.  Train one of the SVM models using a single core, 2 cores, and 4 cores. Compare the speedup (if any).

2.  You might want to look at `rbenchmark` or `microbenchmark` packages for timing.

<!-- -->

1.  What are the training and test error rates? (d) Use the tune() function to select an optimal cost. Consider values in the range 0.01 to 10. (e) Compute the training and test error rates using this new value for cost. (f) Repeat parts (b) through (e) using a support vector machine with a radial kernel. Use the default value for gamma. (g) Repeat parts (b) through (e) using a support vector machine with a polynomial kernel. Set degree=2. (h) Overall, which approach seems to give the best results on this data?
