library(tidyverse)
library(stringr)
library(caret)
library(matrixStats)  
library(rpart)
library(ModelMetrics) 
library(data.table)
library(httr)
library(corrplot)
library(pls)

# Download the MIFoS Dataset
importedData <- read.csv("https://raw.githubusercontent.com/njpatter/mifosdata/master/DataSet_50PixelHeight.txt", header = FALSE)
importedHeaders <- read.csv("https://raw.githubusercontent.com/njpatter/mifosdata/master/DataSet_50PixelHeight_Headers.txt", header = FALSE)

# Apply headers to imported data
colnames(importedData) <- t(importedHeaders)

# There are a lot of NaN values in this data, let's see how many:
print(paste("There are", sum(is.na(importedData)), "NaN values in the imported data"))
# Wow, just over 1/2 of the data is NaN entries (2991130 to be specific)

# Most of these NaN values are expected and represent a lack of exposed pixel faces on the target
# geometry.  However, all of the NaN values in the final drag delta categories are for rows of data
# that contain no useful information (failed or cancelled simulations)
# Let's get rid of those before splitting into train/test and validation sets
importedDataReduced <- importedData %>% filter(!is.na(`dragDeltas-TotalX`))

# Then let's plot to see what the differences in drag forces are across the entire dataset
importedDataReduced %>% 
  ggplot(aes(`dragDeltas-TotalX`)) + 
  geom_histogram()+ 
  scale_y_log10()+ 
  ggtitle("Simulation Drag Distribution") +
  labs(x = "X-Direction Drag Force", y = "Simulation Counts" )  

# There appears to be exceptionally large dragDelta values. Each geometry was only changed by a single
# pixel with dimensions of 2% of the shapes characteristic height.  From a fluid dynamics point 
# of view, this small of a change at the given flow conditions (Re = 1) should not account for a large 
# change in drag value.  Let's cap the drag-delta values at 15% of the average total
# drag force for the original geometries. This is a value that would be unexpectedly large,
# but potentially (fluids pun) not impossible.   
ogDragMean <- 0.15 * mean(importedDataReduced$`originalGeometryDrag-TotalX`)
fluidIntuitionDataReduced <- importedDataReduced %>% 
  filter(abs(`dragDeltas-TotalX`) < ogDragMean)


fluidIntuitionDataReduced %>% 
  ggplot(aes(`dragDeltas-TotalX`)) + 
  geom_histogram() + 
  scale_y_log10()+ 
  ggtitle("Simulation Drag Distribution - Outliers Removed by Fluid Intuition") +
  labs(x = "X-Direction Drag Force", y = "Simulation Counts" ) 

# A little better, but how does this compare to using a data-oriented approach instead of 
# fluid mechanics intuition?

# This time let's use the IQR rule to get rid of outliers in the dragDeltas column
# Specifically, 
# First we find the IQR and the lower/upper quantiles (25%, 75%)
importedIqr <- iqr(importedDataReduced$`dragDeltas-TotalX`) 
lowerQuantile <- quantile(importedDataReduced$`dragDeltas-TotalX`)[2]
upperQuantile <- quantile(importedDataReduced$`dragDeltas-TotalX`)[4]

# Then we calculate the bounds
lowerBound <- lowerQuantile - importedIqr * 1.5 
upperBound <- upperQuantile + importedIqr * 1.5 
# and filter the data based on these bounds
dataDrivenDataReduced <- importedDataReduced %>% 
  filter(`dragDeltas-TotalX` > lowerBound, `dragDeltas-TotalX` < upperBound)

dataDrivenDataReduced %>% 
  ggplot(aes(`dragDeltas-TotalX`)) + 
  geom_histogram() + 
  scale_y_log10()+ 
  ggtitle("Simulation Drag Distribution - Outliers Removed With 1.5 x IQR Approach") +
  labs(x = "X-Direction Drag Force", y = "Simulation Counts" ) 

# Where the Fluid-Intuition approach only removed ~20 outliers, the data driven approach
# removed almost 3,000. Plotting without the log scale shows that the reduced data fits an obvious normal distribution. 
# While the IQR approach definitely provides better looking data, the Fluids Intuition approach
# keeps values that would unexpected, yet possible - The Train/Test and Validation sets and future
# data will be filtered using the Fluids Intuition approach to identifying erroneous simulations.

# The only thing we need to do before creating training, testing, and validation sets is to
# get rid of a few columns that are not relevant to the current analysis.  Specifically,
# all of the dragDelta values except `dragDeltas-TotalX`.  These drag/lift values are also
# outputs of each simulation and using this data would effectively be cheating at data science.
# Additionally, let's get rid of data columns that we know are not going to be useful or
# will only add noise and complicate the analysis. 
RemoveUnusableColumns <- function(dataSet) {
  dataSet %>%
    mutate(dragDeltaX = `dragDeltas-TotalX`) %>%
    select(-starts_with("dragDeltas"),
           # First up, the X and  Y face area calculations
           # These are values that were calculated based on the lines used for integration
           # so they do not contain any simulation related data and may have a negative
           # effect on the analysis.
           -ends_with("face-area"),
           # We also don't need any of the Z-direction forces - all should be zero values
           # anyway because the simulations are all 2D in the x-y plane
           -`originalGeometryDrag-PressureZ`, 
           -`originalGeometryDrag-ViscousZ`, 
           -`originalGeometryDrag-TotalZ`,
           # Lastly, the Geometry, Angle, and Pixel case numbers are only for identification
           # by the user (if it will ever be needed).  The order/distribution of these numbers
           # would have no impact on an arbitrary geometry.
           -`Geometry-Case`,
           -`Angle-Case`,
           -`Pixel-Case`) 
  
}

fluidIntuitionDataReduced <- RemoveUnusableColumns(fluidIntuitionDataReduced)   

# Train, Test, Validation set creation ------------------- 
# Using 10% as a validation set
val_index <- createDataPartition(y = fluidIntuitionDataReduced$dragDeltaX, 
                                 times = 1, p = 0.2, list = FALSE, groups = 200)
trainAndTest <- fluidIntuitionDataReduced[-val_index,]
validationSet <- fluidIntuitionDataReduced[val_index,]

# Using 20% of remaining for a test set
test_index <- createDataPartition(y = trainAndTest$dragDeltaX, 
                                  times = 1, p = 0.2, list = FALSE, groups = 200)
trainSet <- trainAndTest[-test_index,]
testSet <- trainAndTest[test_index,]

# Let's make sure that our splits didn't provide sets weighted one way or another for drag data
setCompare <- data.frame(DataSet = c(rep('Original', nrow(fluidIntuitionDataReduced)), 
                                     rep('Validation', nrow(validationSet)), 
                                     rep('Train', nrow(trainSet)), 
                                     rep('Test', nrow(testSet))), 
                         drag = c(fluidIntuitionDataReduced$dragDeltaX,
                                  validationSet$dragDeltaX, 
                                  trainSet$dragDeltaX, 
                                  testSet$dragDeltaX))
setCompare %>% ggplot(aes(x = drag)) + 
  geom_density(alpha=0.25)+ 
  ggtitle("Dataset Drag Distributions") +
  labs(x = "X-Direction Drag Force", y = "Simulation Density" )  + 
  facet_wrap(.~DataSet)

# The plot variables are overlapping, to make sure we can also compare mean/median/sd
dataSubsetComparison <- setCompare %>% 
  group_by(DataSet) %>%
  summarize(Observations = n(), Mean = mean(drag), Median = median(drag), Sd = sd(drag))
dataSubsetComparison
#########################################
################## - We've got datasets that have similar distributions!  
##################   We can finally get to the data analysis part
#########################################

#########################################
# Data analysis ----------------
# Now that  we've reduced the dataset to usable drag values and split for training, testing,
# and validation, we can proceed with an analysis of the data and any relationships contained therein.
#########################################
 
# Let's take a look at what we need to beat in terms of RMSE - How does the average value perform as a predictor?
mu <- mean(trainSet$dragDeltaX, na.rm = TRUE)
stdDev <- sd(trainSet$dragDeltaX, na.rm = TRUE)
meanTest <- trainSet %>%
  mutate(mu = mu, normDragDelta = (dragDeltaX - mu) / stdDev)
norm_mu <- mean(meanTest$normDragDelta, na.rm = TRUE)
meanTest <- meanTest %>%
  mutate(normMu = norm_mu)
predictionToBeat <- rmse(meanTest$dragDeltaX, meanTest$mu) 
norm_predictionToBeat <- rmse(meanTest$normDragDelta, meanTest$normMu) 


# Let's make a copy of the trainSet just in case we make a mistake along the way
trainSetWithModels <- trainSet
# And subtract the mean (mu) from the dragDelta values
trainSetWithModels <- trainSetWithModels %>%
  mutate(dragDeltaX = dragDeltaX - mu)


# The data from these simulations is coupled - the governing equations of fluid mechanics tell us so... We can't
# decouple most of these relationships. To start we are just going to examine the impact of each variable on 
# drag deltas
# Let's take a look at correlation between drag values and all of the other variables

PlotDragCorrelations <- function(mData, magnitudeFilter = 0.0) {
  corMat <- cor(mData, use ="pairwise.complete.obs", method = "pearson")
  #corrplot(corMat, type="lower")
  pCor <- data.frame(cor = corMat[nrow(corMat),1:ncol(corMat)], var = colnames(mData)) %>% filter(var != "dragDeltaX")
  pCor <- pCor %>% 
    filter(abs(cor) > magnitudeFilter) 
  corMat <- cor(mData, use ="pairwise.complete.obs", method = "spearman") 
  sCor <- data.frame(cor = corMat[nrow(corMat),1:ncol(corMat)], var = colnames(mData)) %>% filter(var != "dragDeltaX")
  sCor <- sCor %>% 
    filter(abs(cor) > magnitudeFilter) 
  
  combined <- pCor %>% filter(var %in% sCor$var) %>% mutate(pearson = abs(cor)) %>% select(-cor)
  combined <- combined %>% left_join(sCor, by = "var") %>% mutate(spearman = abs(cor)) %>% select(-cor)
  p <- combined %>%
    ggplot(aes(x = reorder(var, pearson) )) + 
    geom_point(aes(y = pearson, color = "Pearson")) + 
    geom_point(aes(y = spearman, color = "Spearman"))+ 
    ggtitle("X-Dir Drag - Surface Data Correlation") +
    labs(x = "Surface Data Variable", y = "Correlation Coef." ) +
    theme(axis.text.x = element_text(angle = 90))  
  print(p) 
  combined
}
trainSet_preModels <- trainSetWithModels
dataCorrelations <- PlotDragCorrelations(trainSetWithModels, magnitudeFilter = 0.2)
 
# The correlations between the surface data and drag deltas could be more
# complex than what the Pearson and Spearman correlations can show, but
# the overlap between them suggests there's enough to start creating a model

# Creating a function to simplify calls for plotting relationships to drag
PlotVariableVsDrag <- function(dataset, varName, xLab = "Surface Data Variable") {
  varName <- sym(varName)
  dataset %>% 
    ggplot(aes(x = !!varName, y = dragDeltaX)) + 
    geom_smooth(method = "lm", formula = y ~ poly(x,5)) + #x + I(x^2)+ I(x^3)  ) +  
    geom_point() +
    labs(x = xLab, y = "Drag Force Delta" )
}
# After testing various models, it was found that a 5th order polynomial was
# the best option based on R^2 values

# Let's start by visualizing the largest magnitude correlation - derivative of the x-velocity
# with respect to the x-direction: du/dx for face 2 (dx-velocity-dx)

PlotVariableVsDrag(trainSetWithModels, "faceData-2-dx-velocity-dx", "Face 2: du/dx")  
 

# Train the lm model for Face2 du-dy 
# Let's make a reusable function for creating models & simplify future calls

CreateModel <- function(datSet, varName ) {
  varName <- sym(varName)
  train( 
    form = y ~ poly(x,5), #x + I(x^2) + I(x^3) ,
    data = datSet %>%
     filter(!is.na(!!varName)) %>%
     select(x = !!varName,
            y = dragDeltaX),
    method = 'lm'
  )
}

# Then we use the function to create the model
model_f2_du_dx <- CreateModel(trainSetWithModels, "faceData-2-dx-velocity-dx")
summary(model_f2_du_dx)

# Create prediction weights based on the model
trainSetWithModels$weight_f2_du_dx = predict(model_f2_du_dx, 
                                 newdata = data.frame(
                                   x = trainSetWithModels$`faceData-2-dx-velocity-dx`),
                                 na.action = na.pass)
# Calculate our new set of predictions based on Mean and F2-du-dx weight
trainSetWithModels <- trainSetWithModels %>%  
  mutate(pred = mu + ifelse(!is.na(weight_f2_du_dx), weight_f2_du_dx, 0))

# Calculate our RMSE for just the training set at this point
weightOneRmse <- rmse(trainSet$dragDeltaX, trainSetWithModels$pred )  
print(paste("Improvement over mean value prediction: ", 
            (100*(predictionToBeat - weightOneRmse) / predictionToBeat), "%"))

# Modify dragDelta values to remove the effect of the first variable weight
trainSetWithModels <- trainSetWithModels %>%
  mutate(dragDeltaX = dragDeltaX + ifelse(!is.na(weight_f2_du_dx), - weight_f2_du_dx, 0))




# Visualize the largest correlations and use them to pick the next variable
trainSet_modelOne <- trainSetWithModels
dataCorrelations <- PlotDragCorrelations(trainSetWithModels %>% select(-pred, -weight_f2_du_dx), 
                                         magnitudeFilter = 0.2)




# Next most useful variable is Face-4 du/dx (faceData-4-dx-velocity-dx) - 
# makes sense that it is the same variable because Face 2 and Face 4 are 
# opposite sides of the pixel
# Visualize first
PlotVariableVsDrag(trainSetWithModels, "faceData-4-dx-velocity-dx") 

# The same 5th order polynomial works well for this variable, so next up
# is training the model 
model_f4_du_dx <- CreateModel(trainSetWithModels, "faceData-4-dx-velocity-dx")
summary(model_f4_du_dx)

# Use the model to make predictions
trainSetWithModels$weight_f4_du_dx = predict(model_f4_du_dx, 
                                 newdata = data.frame(
                                   x = trainSetWithModels$`faceData-4-dx-velocity-dx`),
                                 na.action = na.pass)
# Calculate the new predictions using the mean and both weights
trainSetWithModels <- trainSetWithModels %>% 
  mutate(pred = mu + 
           ifelse(!is.na(weight_f4_du_dx), weight_f4_du_dx, 0) +
           ifelse(!is.na(weight_f2_du_dx), weight_f2_du_dx, 0))

# Calculate the new RMSE based on two weights
weightTwoRmse <- rmse(trainSet$dragDeltaX, trainSetWithModels$pred )  
print(paste("Improvement over mean value prediction: ", 
            (100*(predictionToBeat - weightTwoRmse) / predictionToBeat), "%"))      

# Remove the effect of the new weight 
trainSetWithModels <- trainSetWithModels %>%
  mutate(dragDeltaX = dragDeltaX + ifelse(!is.na(weight_f4_du_dx), -weight_f4_du_dx, 0)) 





# Examine correlation coefficients without the weights
dataCorrelations <- PlotDragCorrelations(trainSetWithModels %>% 
                                           select(-pred, -weight_f2_du_dx, -weight_f4_du_dx), 
                                         magnitudeFilter = 0.075)




# The following plots and variables chosen depend heavily on the train/test/validation splits
# All of the variables tested exhibit the same flat, zero-centric distributions or
# are discrete in nature (2-3 vertical lines of points)

# Next, we test some of the variables with the largest coefficients (at for some of the earlier
# data splits)
PlotVariableVsDrag(trainSetWithModels, "faceData-2-dp-dx", "Face 2 dp-dx") 
#OOF, that is a really flat line.  


PlotVariableVsDrag(trainSetWithModels, "faceData-2-x-wall-shear", "Face 2 Wall Shear") 
#OOF, that is another really flat line.  


PlotVariableVsDrag(trainSetWithModels, "faceData-3-dx-velocity-dy", "Face 3 du-dy")
#Still pretty flat 
   
# Unfortunately all of the remaining variables either have flat lines (centered around zero)
# for plotted data or produce models with really poor R^2 values



# Before we move on, let's take a look at how our current options perform
# on the test set (we'll save the validation set testing for our final act)

tempTest <- testSet %>% 
  mutate(pred_mean = mu,
         weight_f2_du_dx = predict(model_f2_du_dx, 
                                  newdata = data.frame(
                                    x = `faceData-2-dx-velocity-dx`),
                                  na.action = na.pass),
         pred_oneWeight = mu + ifelse(!is.na(weight_f2_du_dx), weight_f2_du_dx, 0),
         weight_f4_du_dx = predict(model_f4_du_dx, 
                 newdata = data.frame(
                   x = `faceData-4-dx-velocity-dx`),
                 na.action = na.pass),
         pred_twoWeight = mu + 
           ifelse(!is.na(weight_f4_du_dx), weight_f4_du_dx, 0) +
           ifelse(!is.na(weight_f2_du_dx), weight_f2_du_dx, 0)
         )

# Create a data frame that includes model performance - we'll add the PCR models as they are created
modelPerformance <- data.frame(Models = c("Mean", "Mean + F2 du-dx", "Mean + F2 du-dx + F4 du-dx"),
                                     RMSE = c(rmse(tempTest$dragDeltaX, tempTest$pred_mean),
                                              rmse(tempTest$dragDeltaX, tempTest$pred_oneWeight),
                                              rmse(tempTest$dragDeltaX, tempTest$pred_twoWeight)
                                              ))
meanRmse <- rmse(tempTest$dragDeltaX, tempTest$pred_mean)
# Calculate relative performance - as improvement from mean prediction RMSE
modelPerformance <- modelPerformance %>%
  mutate(PercentImprovement = (meanRmse - RMSE) / meanRmse * 100)

###################################################################################
###################################################################################
# Time for approach #2 -----
# Principal Component Regression
### 
# Next up is attempting to use PCR to *hopefully* get better prediction performance
# than the two model method from part 1 of the analysis
# To do this we can't have NA values (or at least can't come up with anything
# useful with NA values in every row)
# Let's scale the data (mean-center and var = 1)

# We need means and sd's from each column to center and scale the data
trainMeans <- apply(trainSet, 2, mean, na.rm = TRUE)
trainStdDev <- apply(trainSet, 2, sd, na.rm = TRUE)
dragMean <- trainMeans[length(trainMeans)]
dragStdDev <- trainStdDev[length(trainStdDev)]

# Create a re-usable function to center and scale data
MeanCenterData <- function(dataSet, colMeans, colStdDev) {
  # Mean-center 
  tempSet <- sweep(dataSet, 2, colMeans, "-")
  # & scale data columns
  tempSet <- sweep(tempSet, 2, colStdDev, "/") 
  
  # Then fill in the NA values with zero
  tempSet[is.na(tempSet)] <- 0
  tempSet
}

# And simplify how we are going to create PCR models
CreatePcrModel <- function(dataSet, colMeans, colStdDev) {
  # Copy the data set because I'm paranoid
  tempSet <- dataSet
  # Mean-center and scale the data
  tempSet <- MeanCenterData(tempSet, colMeans, colStdDev)
  # Train the PCR model
  pmod <- train(dragDeltaX~., 
                    data = tempSet,
                    method = "pcr", 
                    trControl = trainControl("cv", number = 10),
                    tuneLength = 110)  
  # Return the model
  pmod
}

# We can then train the PCR model
pcrModel <- CreatePcrModel(trainSet, trainMeans, trainStdDev)
# And visualize the improvements provided by the PCs
plot(pcrModel)


# To determine performance on the test set
# first we need to normalize the test data as we did for the train data (same mean/sd values)
PredictDrag <- function(aModel, dataSet, colMeans, colStdDev) {
  tempSet <- MeanCenterData(dataSet, colMeans, colStdDev) 
  
  # Get rid of the dragDelta column 
  tempSet <- tempSet %>%
    select(-dragDeltaX)
  
  predictions <- predict(aModel, tempSet)
}

# Then use the function to calculate drag predictions for the test set
pcrPredictions <- PredictDrag(pcrModel, testSet, trainMeans, trainStdDev)

# and then use the normalized drag values output from the PCR model to 
# calculate what the actual drag predictions would be
pcrPredictions <- pcrPredictions * dragStdDev + dragMean

# and use those values to calculate the test RMSE
pcrRmse <- rmse(testSet$dragDeltaX, pcrPredictions)

# Then we can add this new model performance to our ongoing data frame
modelPerformance <- bind_rows(modelPerformance, 
          data.frame(Models = "PCR",
                     RMSE = pcrRmse,
                     PercentImprovement = 100 *(meanRmse - pcrRmse) /meanRmse))


# Too much time was spent creating this dataset to not know what
# is driving the PCR model.  Let's try to figure out what the principal 
# components are representing.
pc1 <- data.frame(x = names(pcrModel$finalModel$loadings[,1]), 
                     y = pcrModel$finalModel$loadings[,1])
pc1 %>% filter(abs(y) > 0.125) %>%
  ggplot(aes(x = reorder(x, abs(y)), abs(y))) + 
  geom_point()+
  labs(x = "Features", y = "PC Loadings" ) +
  theme(axis.text.x = element_text(angle = 90))  

# The loadings for PC1 incorporate whether the pixel is added or subtracted from
# the original shape, but most of the larger loadings appear to be focused on
# the data on face 4.
# Let's take a look at PC2

pc2 <- data.frame(x = names(pcrModel$finalModel$loadings[,2]), 
                  y = pcrModel$finalModel$loadings[,2])
pc2 %>% filter(abs(y) > 0.125) %>%
  ggplot(aes(x = reorder(x, abs(y)), abs(y))) + 
  geom_point()+
  theme(axis.text.x = element_text(angle = 90)) 

# The loadings for PC2 also focus on face 4, just different variables

pc3 <- data.frame(x = names(pcrModel$finalModel$loadings[,3]), 
                  y = pcrModel$finalModel$loadings[,3])
pc3 %>% filter(abs(y) > 0.125) %>%
  ggplot(aes(x = reorder(x, abs(y)), abs(y))) + 
  geom_point()+
  theme(axis.text.x = element_text(angle = 90)) 

# The loadings for PC3 are a wider mixture (fluids pun!) of variables
# involving original geometry drag values and data from faces 4, 3, and 1

#Looking back at the PCR - RMSE plot, I find it interesting that the initial
# components are not the ones that provide the largest decreases in RMSE.
# Let's take a look at one responsible for a large drop - PC19

pc19 <- data.frame(x = names(pcrModel$finalModel$loadings[,19]), 
                  y = pcrModel$finalModel$loadings[,19])
pc19 %>% filter(abs(y) > 0.125) %>%
  ggplot(aes(x = reorder(x, abs(y)), abs(y))) + 
  geom_point()+
  theme(axis.text.x = element_text(angle = 90)) 
 

# Another such drop occurs for PC70... let's take a look

pc70 <- data.frame(x = names(pcrModel$finalModel$loadings[,70]), 
                   y = pcrModel$finalModel$loadings[,70])
pc70 %>% filter(abs(y) > 0.125) %>%
  ggplot(aes(x = reorder(x, abs(y)), abs(y))) + 
  geom_point()+
  theme(axis.text.x = element_text(angle = 90)) 

 

###################################################################################
# While thinking about the PCs representing various versions of pressure and viscous
# drag, I realized that there may be a fundamental difference in principal components
# between adding and subtracting pixels.
#
# Principal Component Regression - Additive vs Subtractive Approach

# Additive side:
# Create the additive subset from the train set
additiveTrainSet <- trainSet %>% 
  filter(isPixelAdded == 1) %>%
  select(-isPixelAdded, -isPixelSubtracted)

# Calculate all necessary column stats to mean-center and scale prior to creating PCR model
addMeans <- apply(additiveTrainSet, 2, mean, na.rm = TRUE)
addStdDev <- apply(additiveTrainSet, 2, sd, na.rm = TRUE)
addDragMean <- addMeans[length(addMeans)]
addDragStdDev <- addStdDev[length(addStdDev)]

# Train the PCR additive model
additivePcr <- CreatePcrModel(additiveTrainSet, addMeans, addStdDev)
# Plot RMSE over components
plot(additivePcr)

# Great, this shows an even better drop in RMSE!  Let's do the same for the subtractive side

# Subtractive side:
# Create the subtractive subset from the train set
subtractiveTrainSet <- trainSet %>% 
  filter(isPixelSubtracted == 1) %>%
  select(-isPixelAdded, -isPixelSubtracted)

# Calculate all necessary column stats to mean-center and scale prior to creating PCR model
subMeans <- apply(subtractiveTrainSet, 2, mean, na.rm = TRUE)
subStdDev <- apply(subtractiveTrainSet, 2, sd, na.rm = TRUE)
subDragMean <- addMeans[length(subMeans)]
subDragStdDev <- addStdDev[length(subStdDev)]

# Train the PCR subtractive model
subtractivePcr <- CreatePcrModel(subtractiveTrainSet, subMeans, subStdDev)
# Plot RMSE over components
plot(subtractivePcr)

# Also better than the original PCR model.  Let's try it out on the test data

# Separate out the additive/subtractive subsets
additiveTestSet <- testSet %>% 
  filter(isPixelAdded == 1) %>%
  select(-isPixelAdded, -isPixelSubtracted)
subtractiveTestSet <- testSet %>% 
  filter(isPixelSubtracted == 1) %>%
  select(-isPixelAdded, -isPixelSubtracted)

# Predict normalized drag values for test set
addPred <- PredictDrag(additivePcr, additiveTestSet, addMeans, addStdDev)
subPred <- PredictDrag(subtractivePcr, subtractiveTestSet, subMeans, subStdDev)

# De-normalize? Un-normalize? Ab-normalize? you get the idea... return normalized
# drag values to actual drag value predictions
addPred <- addPred * addDragStdDev + addDragMean
subPred <- subPred * subDragStdDev + subDragMean

# Combine predictions and actual drag values into appropriately arranged vectors
allPred <- c(addPred, subPred)
allDrag <- c(additiveTestSet$dragDeltaX, subtractiveTestSet$dragDeltaX)

# Calculate RMSE (not RSME as the variable name suggests)
additiveRSME <- rmse(addPred, additiveTestSet$dragDeltaX)
subtractiveRSME <- rmse(subPred, subtractiveTestSet$dragDeltaX)

# Calculate the combined RMSE
addSubPcrRmse <- rmse(allDrag, allPred)

# Add the split PCR approach to the model performance data frame
modelPerformance <- bind_rows(modelPerformance, 
                                    data.frame(Models = "Add/Sub Split PCR",
                                               RMSE = addSubPcrRmse,
                                               PercentImprovement = 100 *(meanRmse - addSubPcrRmse) /meanRmse))

# Display the model performance data frame
modelPerformance

#### All code for comparing model performance on the Validation set was included in the Rmd file. This comparison
# was completed while writing the results and conclusions sections of the analysis

# If you want to run the Rmd file, you need to run this script first - 
# it saves the workspace so that it can be loaded by the Rmd file
save.image(file='MIFoS - Data Analysis Workspace.RData')