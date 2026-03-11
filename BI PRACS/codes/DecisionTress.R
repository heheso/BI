# 1. Install and load the required package
# Run install.packages("party") once if it's not in your lab
library(party)

# 2. Prepare the input data
# We take the first 105 rows of the built-in readingSkills dataset
input.dat <- readingSkills[c(1:105), ]
print(input.dat)

# 3. Open a PNG file to save the visualization
png(file = "decision_tree.png")

# 4. Create the Decision Tree Model
# Format: Target ~ Feature1 + Feature2 + Feature3
output.tree <- ctree(
  nativeSpeaker ~ age + shoeSize + score, 
  data = input.dat
)

# 5. Plot and Save
plot(output.tree)
dev.off()
