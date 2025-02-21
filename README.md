# Predicting-NFL-Game-Outcomes
Predicting NFL game outcomes using Logistic Regression and evaulating models using ROC and AUC
Hunter Grigsby

- The binary outcome (y-variable) for predicting is the result of the game (0=loss, 1=win).

- The predictor variables are Yard Margin, 3rd down conversion rate, Home/Away, penalties, and 3rd down conversion rate (3=high, 2=moderate, 1=low).

- A win indicates whether the team won a game, with 1 representing a win and 0 a loss. 'Yards' refers to the total yards gained by the team in each game, while 'Yards Allowed' represents the total yards given up to the opposing teams. 'Yard Margin' is the difference between yards gained and yards allowed. The '3rd Down Conversion Rate' measures the team's performance on third downs, rated on a scale of 1 to 3, with 1 being the worst, 2 average, and 3 the best. 'Points' denote the team's total score in each game, while 'Points Allowed' indicates the points scored by the opposition. 'Point Margin' is the difference between points scored and points allowed. 'First Downs' represents the number of first downs the team achieved per game. 'Penalties' are the total penalties committed by the team. 'Home and Away' specifies the location of the game, with 1 for home and 2 for away.

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)

ggplot2::theme_set(ggplot2::theme_light())

knitr::opts_chunk$set(
  fig.align = "center",
  out.width = "70%"  # Scale figures to take 70% of slide width
)
```
# Data Peparation
```{r}
# Helper packages
library(dplyr)     # for data wrangling
library(ggplot2)   # for plotting
library(rsample)   # for data splitting

# Modeling packages
library(caret)     # for logistic regression modeling

# Model interpretability packages
# Install 'vip' only if it is not installed
if (!requireNamespace("vip", quietly = TRUE)) {
  install.packages("vip")
}
library(vip)       # variable importance

# Evaluate model using ROC and AUC
library(pROC)
library(ROCR)

file.exists("data_stats_362_NFL2.csv")
NFL2 <- data_stats_362_NFL2

# Initial dimension
dim(NFL2)

# Response variable: Display the first few entries of 'WIN'
head(NFL2$WIN)

# Convert target variable 'WIN' to a binary factor
NFL2$WIN <- as.factor(ifelse(NFL2$WIN == TRUE, 1, 0))
NFL2$`3RD-DOWN-CONV-RATE` <- as.factor(NFL2$`3RD-DOWN-CONV-RATE`)
NFL2$`HOME/AWAY` <- as.factor(NFL2$`HOME/AWAY`)
```
- I am currently creating my own dataset, collecting variables and data from Pro Football Reference. I organize the dataset in Google Sheets, which includes NFL statistics from the current NFL season (2024-2025, first 3 weeks), covering 96 games.

- To include variables such as point margin and yard margin, I create these by subtracting specific variables. For example, to calculate the point margin, I subtract the points allowed from the points the team scored.

- I prepare the dataset using data from this current NFL season, sourced from Pro Football Reference. I input the data and develop new variables like yard margin and point margin. I also categorize the third-down conversion rate into a high, medium, and low rating system.

## Examine the Distribution of "WIN"
```{r}
summary(NFL2$WIN)
```

# Traditional Logistic Regression
```{r}
LR <- glm(formula = WIN ~ `YARD MARGIN` + `3RD-DOWN-CONV-RATE` + `HOME/AWAY` + PENALTIES,
          family = "binomial", data = NFL2)
summary(LR)

# Extract coefficients
coef(LR)

# Calculate Odds Ratios (ORs)
ORs = exp(coef(LR))
CIs = exp(confint(LR))
cbind(ORs, CIs)
```
## Interpretation of Odds Ratios
```{r}
# Predict probabilities on the same dataset
traditional_probs <- predict(LR, type = "response")

# Convert probabilities to binary predictions
traditional_pred <- ifelse(traditional_probs > 0.5, 1, 0)

# Convert predictions and actual values to factors with matching levels
traditional_pred <- factor(traditional_pred, levels = c(0, 1))
NFL2$WIN <- factor(NFL2$WIN, levels = c(0, 1))

# Calculate confusion matrix
confusion_matrix_traditional <- confusionMatrix(traditional_pred, NFL2$WIN)
confusion_matrix_traditional
```
## Evaulate model using ROC and AUC
```{r}
# Evaluate model using ROC and AUC
roc_curve_traditional <- roc(NFL2$WIN, traditional_probs)
auc_traditional <- auc(roc_curve_traditional)
auc_traditional

# Plot ROC curve
plot(roc_curve_traditional, col = "blue", 
     main = "ROC Curve for Traditional Logistic Regression")
```
# Logistic Regression with Machine Learning
## Train vs Test Split
```{r}
# Split the NFL data into training (70%) and test (30%) sets
set.seed(123)  # for reproducibility
nfl_split = initial_split(NFL2, 
                          prop = 0.7, 
                          strata = "WIN")
nfl_train = training(nfl_split)
nfl_test = testing(nfl_split)
```
# Logistic Regression with Train/Test Split
```{r}
model1 <- glm(WIN ~ `YARD MARGIN` + `3RD-DOWN-CONV-RATE` + `HOME/AWAY` + PENALTIES, 
              family = "binomial", data = nfl_train)
summary(model1)
```
```{r}
# Convert coefficients to odds ratios
cbind(ORs = exp(coef(model1)), exp(confint(model1)))
```
## Predicited Probabilities
```{r}
# Predict probabilities on the test set
ml_probs <- predict(model1, newdata = nfl_test, type = "response")

# Convert probabilities to binary predictions
ml_pred <- ifelse(ml_probs > 0.5, 1, 0)

# Convert predictions and actual values to factors with matching levels
ml_pred <- factor(ml_pred, levels = c(0, 1))
nfl_test$WIN <- factor(nfl_test$WIN, levels = c(0, 1))

# Calculate confusion matrix
confusion_matrix_ml <- confusionMatrix(ml_pred, nfl_test$WIN)
confusion_matrix_ml
```
```{r}
# Evaluate ML logistic model using ROC and AUC
roc_curve_ml <- roc(nfl_test$WIN, ml_probs)
auc_ml <- auc(roc_curve_ml)
auc_ml

# Plot ROC curve
plot(roc_curve_ml, col = "red", 
     main = "ROC Curve for Logistic Regression (Machine Learning)")
```
# Model Comparison
```{r}
# AIC for the traditional logistic regression model
aic_traditional <- AIC(LR)
print(aic_traditional)

# AIC for the logistic regression model with train-test split
aic_ml <- AIC(model1)
print(aic_ml)
```
## Conclusion
- The machine learning approach using the train/test split performed better than the traditional logistic regression model, as indicated by a potentially higher AUC and a lower AIC. This improvement is due to the machine learning model’s better handling of overfitting by evaluating performance on unseen test data. Although regularization was not explicitly applied in this analysis, the split strategy itself enhanced the model's predictive accuracy.

- A key limitation of this analysis is the limited set of features used, potentially missing other influential factors like team-specific statistics. Additionally, no regularization was employed to address multicollinearity or overfitting. Future improvements could involve adding more features, applying regularization techniques, and using more advanced models like Random Forests or cross-validation to refine predictive accuracy.








