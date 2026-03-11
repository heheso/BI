# [1] Importing the Dataset
# Note: Only run install.packages() once on your lab computer!
# install.packages("dplyr") 
library(dplyr)

# Look at the first few rows of the data
head(mtcars)

# [2] Splitting the Dataset
# install.packages("caTools") 
library(caTools)

# Split data into 80% for training and 20% for testing
split <- sample.split(mtcars, SplitRatio = 0.8)
train_reg <- subset(mtcars, split == "TRUE")
test_reg <- subset(mtcars, split == "FALSE")

# [3] Building the model
# Train the model using features in the dataset
logistic_model <- glm(vs ~ wt + disp, data = train_reg, family = "binomial")

# View the model and its summary statistics
logistic_model
summary(logistic_model)

# [4] Predicting on Unseen Data
# We use the test_reg dataset we created in Step 2.
# type = "response" tells R to output probabilities (between 0 and 1).
predict_reg <- predict(logistic_model, test_reg, type = "response")

# View the raw probabilities 
print(predict_reg)

# Convert probabilities into binary classes (0 or 1)
# If the probability is greater than 50% (0.5), classify as 1. Otherwise, 0.
predicted_classes <- ifelse(predict_reg > 0.5, 1, 0)

# Compare our predictions against the actual data
print(predicted_classes)