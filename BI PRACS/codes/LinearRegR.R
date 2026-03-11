# [1] Create Relationship Model & get the Coefficients
# Values of height (Predictor)
x <- c(151, 174, 138, 186, 128, 136, 179, 163, 152, 131)
# Values of weight (Response)
y <- c(63, 81, 56, 91, 47, 57, 76, 72, 62, 48)

# Apply the lm() function
relation <- lm(y~x)
print(relation)

# [2] Get the Summary of the Relationship
# This shows the average error in prediction, also called residuals
print(summary(relation))

# [3] Predict the weight of new persons
# Find weight of a person with height 170
a <- data.frame(x = 170)
result <- predict(relation,a)
print(result)

# [4] Visualize the Regression Graphically
# Give the chart file a name
png(file = "linearregression.png")

# Plot the chart
plot(y,x,col = "blue",main = "Height & Weight Regression",
     cex = 1.3,pch = 16,xlab = "Weight in Kg",ylab = "Height in cm")
abline(lm(x~y))

# Save the file
dev.off()