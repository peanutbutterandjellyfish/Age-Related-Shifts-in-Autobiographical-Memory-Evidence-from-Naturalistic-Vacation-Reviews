# AUTHOR: Flavia Tinner

# DESCRIPTION:
# This R code implements the analysis described in the paper 
# "Age-Related Shifts in Autobiographical Memory: Evidence from Naturalistic Vacation Reviews."
# The code analyzes a dataset of 377,477 German vacation reviews, investigating age-related changes in autobiographical memory.
# The study employs a novel, fully automated linguistic method to quantify episodic and semantic content in naturalistic contexts. 
# The code includes the analysis for operationalizing episodic memory through word count and semantic memory through average word length,
# revealing how age-related shifts in memory detail—specifically a decrease in episodic content and an increase in 
# semantic abstraction—mirror findings from lab-based autobiographical memory studies.

# Load required packages
packages <- c("doc2concrete", "tidyverse") # https://cran.r-project.org/web/packages/doc2concrete/doc2concrete.pdf

# Check and install missing packages
package.check <- lapply(
  packages,
  FUN = function(x) {
    if (!require(x, character.only = TRUE)) {
      install.packages(x, dependencies = TRUE)
      library(x, character.only = TRUE)
    }
  }
)

# Clear Workspace
rm(list = ls())

# Set the working directory to the location where the R code and data files are stored.
setwd("C:/Users/TinnerF/Dropbox/RCode/Aging")

# Import Dictionaries
# The AAIV (Automatisch generierte Normen für Abstrakheit, Arousal, Vorstellbarkeit und Valenz) dictionary
# is a resource containing 350,000 German lemmatised words, rated on four psycholinguistic attributes:
# Abstractness, Arousal, Imageability, and Valence. These ratings were generated via a supervised learning algorithm.
# - Abstractness measures whether a word refers to something abstract (e.g., "idea") or concrete (e.g., "image").
# - Arousal measures the emotional intensity of a word (e.g., "bestialisch" = high arousal, "ausgerollt" = low arousal).
# - Imageability refers to how easily a word conjures a mental image (e.g., "neonerleuchtet" = high imageability).
# - Valence measures the pleasantness of a word (e.g., "wundervoll" = positive, "katastrophenmässig" = negative).
# The dictionary was downloaded from the University of Stuttgart's website (link provided).
# The 'ratings_lrec16_koeper_ssiw.txt' file is loaded into R using fread from the data.table package with UTF-8 encoding.
# The 'Word' column is then converted to lowercase for consistent text processing.
# Authors: Maximilian Köper & Sabine Schulte im Walde (2016)
# Resource: https://www.ims.uni-stuttgart.de/forschung/ressourcen/experiment-daten/affective-norms/
# Paper Reference: "Automatically Generated Affective Norms of Abstractness, Arousal, Imageability and Valence for 350 000 German Lemmas" (Köper & Schulte im Walde, LREC 2016)
# Link to Paper: https://aclanthology.org/L16-1413/

AAIV <- data.table::fread("ratings_lrec16_koeper_ssiw.txt", encoding = "UTF-8")
AAIV$Word <- tolower(AAIV$Word)

################################################################################
# Lemmatization & Part of Speech Tagging
# This section prepares the data by importing it, performing lemmatization, and tagging parts of speech (POS) using the udpipe package.
# However, the POS tagging process can take a long time to generate, so we are reading the pre-annotated output below instead of re-running it.

# Import Data
# Read in the dataset (LIWC output) and filter for reviews where text is provided.
# 'review.provided == 1' filters for reviews that are included, and '!is.na(text)' excludes empty reviews.
df <- data.table::fread("LIWC_Output.csv", encoding = "UTF-8") 
df <- df %>% filter(review.provided == 1 & !is.na(text)) 

# Convert Character Date to Date
# Convert 'reviewDate' from character format to Date format 
# 'reviewMonth' extracts the month from the 'reviewDate' for easier analysis.
df$reviewDate <- as.Date(df$reviewDate, format = "%d/%m/%Y")
df$reviewMonth <- as.numeric(substr(df$reviewDate, start = 6, stop = 7))

# Load Necessary Libraries
# Load the 'udpipe' for lemmatization and POS tagging, 'data.table' for fast data manipulation,
# and 'parallel' for parallel processing of large datasets.
library(udpipe)
library(data.table)
library(parallel)

# Prepare Data for Annotation
# Select relevant columns for annotation (review day, text, and review ID).
df_annotate <- df %>% select(review_day, text, reviewID)

# Load Language Model
# Download and load the pre-trained 'german-gsd' language model for lemmatization and POS tagging.
# This step is crucial for processing the German text data.
ud_model <- udpipe_download_model(language = "german-gsd")
ud_model <- udpipe_load_model(ud_model$file_model)

# Annotate Text Function
# This function takes text from the 'df_annotate' dataframe, applies the udpipe annotation model,
# and returns relevant parts of speech annotations, excluding unnecessary columns.
annotate_splits <- function(x) {
  x <- as.data.table(udpipe_annotate(ud_model, x = x$text,
                                     doc_id = x$reviewID, tagger = "default",
                                     parser = "none"))
  # Remove not required columns to keep the dataset clean and concise.
  x <- x[, c("sentence", "feats", "head_token_id", "dep_rel", "deps") := NULL]
  return(x)
}

# Split the Data for Parallel Processing
# The corpus is split into chunks (every two reviews) for parallel processing to speed up the annotation process.
corpus_splitted <- split(df_annotate, seq(1, nrow(df_annotate), by = 2))

# Parallel Annotation
# The annotation is applied in parallel to each chunk of data using the 'mclapply' function.
# This speeds up the annotation process significantly for large datasets.
annotation <- mclapply(corpus_splitted, FUN = function(x) annotate_splits(x))

# Combine Annotated Data
# After annotation, the results from all chunks are combined into one dataframe using 'rbindlist'.
annotation <- rbindlist(annotation)

# Save Annotated Data
# Save the annotated data to a CSV file so it can be loaded later without re-running the slow annotation process.
# write.csv(annotation, "Annotated_HC_POS.csv")

# Read Pre-annotated Data
# Since POS tagging and lemmatization can take a long time, we use the pre-annotated file for further analysis.
# This avoids the need to re-run the time-consuming annotation process.
POS <- data.table::fread("Annotated_HC_POS.csv", encoding = "UTF-8")


# Convert lemmas to lowercase to ensure consistent casing
# The text data may have mixed casing for the lemmas, so converting all lemmas to lowercase
# ensures consistency and prevents the same word in different cases from being treated as distinct.
POS$lemma <- tolower(POS$lemma)

# Filter out unwanted POS (Part of Speech) types and rows with missing 'upos' values
# We exclude rows where the 'upos' (universal part of speech tag) is one of the following:
# - "NUM" (number)
# - "SCONJ" (subordinating conjunction)
# - "SYM" (symbol)
# - "X" (other)
# - "PUNCT" (punctuation)
# These POS types are not relevant for our analysis, so they are removed from the dataset.
POS <- POS %>% filter(!is.na(upos) & 
                        upos != "NUM" & 
                        upos != "SCONJ" &
                        upos != "SYM" &
                        upos != "X" &
                        upos != "PUNCT")

# Clean lemmas by removing punctuation, numbers, and trimming whitespace
# This cleaning step is crucial to focus on the meaningful words (lemmas) without distractions from 
# irrelevant characters like punctuation or numbers. Any rows where the 'lemma_cleaned' becomes empty 
# after cleaning are filtered out to ensure that only meaningful words are included in the analysis.
POS <- POS %>%
  mutate(
    lemma_cleaned = gsub("[[:punct:]0-9]", "", lemma)  # Remove punctuation and numbers using regex
  ) %>%
  filter(lemma_cleaned != "")  # Remove any rows where the 'lemma_cleaned' is empty after cleaning

# Trim any leading or trailing whitespace from 'lemma_cleaned'
# This step removes any extra spaces that might exist before or after the lemmas to ensure clean data for analysis.
POS$lemma_cleaned <- stringr::str_trim(POS$lemma_cleaned)

##############################################################################
# Compute Word Length for Each Lemma

# Calculate the length (number of characters) of each cleaned lemma using the 'str_length' function
# from the stringr package. This gives an indication of the word's size, which is relevant for analyzing
# how word length correlates with concreteness and imageability ratings. Shorter words are often more concrete,
# while longer words may be more abstract or less imageable, which will be examined in further analysis.
POS$NCHAR <- stringr::str_length(POS$lemma_cleaned)

# Assign Cleaned Lemma to Original Lemma Variable
# After cleaning the lemma (removing punctuation, numbers, and extra spaces), we replace the original 'lemma'
# column with the cleaned version ('lemma_cleaned') for consistency in the dataset.
POS$lemma <- POS$lemma_cleaned

# REMOVE DUPLICATES - NOT REQUIRED FOR POS ANALYSIS ONLY 
# We remove duplicates from the dataset, even though they are not required for the POS tagging analysis.
# It's noted that there are some duplicates in the AAIV dataset from Köper & Schulte im Walde (2016),
# which could affect subsequent analyses, so these are eliminated here to ensure uniqueness.

# Remove duplicate words from AAIV dataset to ensure uniqueness
# The 'duplicated' function identifies and removes rows where the 'Word' column has duplicate entries.
# A new column 'NCHAR' is then created for the unique words to store the length of each word.
AAIV_UNIQUE <- AAIV[!duplicated(AAIV$Word), ]
AAIV_UNIQUE$NCHAR <- stringr::str_length(AAIV_UNIQUE$Word)

# Remove duplicate lemmas from the POS dataset to ensure uniqueness
# Similarly, duplicate cleaned lemmas are removed from the 'POS' dataset to avoid redundant entries.
POS_UNIQUE <- POS[!duplicated(POS$lemma_cleaned), ]

################################################################################
# This section performs descriptive statistics, checks for missing data, and visualizes the relationship between word length and abstraction (concreteness score).
# It also tests the assumptions for linear regression (linearity, homoscedasticity, and normality of residuals) before fitting and summarizing the model.

# DESCRIPTIVE STATISTICS
# Load the 'psych' package for descriptive statistics
# Use the 'describe' function to compute and display summary statistics (e.g., mean, standard deviation) 
# for the 'AbstConc' column (Abstraction/Concreteness score) in the AAIV_UNIQUE dataset.
library(psych)

# Check for missing data
# Before proceeding with any analysis, it's important to check for missing values in the dataset.
# If there are missing values in the variables 'NCHAR' (word length) or 'AbstConc' (abstraction/concreteness score),
# a warning message is displayed to inform the user, but the analysis proceeds after removing the missing data.
if (any(is.na(AAIV_UNIQUE$NCHAR)) || any(is.na(AAIV_UNIQUE$AbstConc))) {
  cat("Warning: Missing values detected. Proceeding after removing missing data.\n")
}

# Remove missing values
# Use 'na.omit' to remove any rows with missing values in the dataset.
# This step ensures that only complete cases are included in the analysis, avoiding potential issues with missing data.
AAIV_clean <- na.omit(AAIV_UNIQUE)

# 1. Visualize distributions for normality with histograms and Q-Q plots
par(mfrow = c(2, 2))  # Arrange plots in a 2x2 grid
hist(AAIV_UNIQUE$IMG, main = "Histogram of IMG", xlab = "IMG", col = "lightblue", border = "white")
hist(AAIV_UNIQUE$AbstConc, main = "Histogram of AbstConc", xlab = "AbstConc", col = "lightgreen", border = "white")
qqnorm(AAIV_UNIQUE$IMG, main = "Q-Q Plot of IMG")
qqline(AAIV_UNIQUE$IMG, col = "red")
qqnorm(AAIV_UNIQUE$AbstConc, main = "Q-Q Plot of AbstConc")
qqline(AAIV_UNIQUE$AbstConc, col = "red")

# 2. Perform Anderson-Darling test for normality (better for large samples)
ad_img <- ad.test(AAIV_UNIQUE$IMG)
ad_abstconc <- ad.test(AAIV_UNIQUE$AbstConc)

# Print Anderson-Darling test results
cat("Anderson-Darling Test for IMG:\n", ad_img, "\n\n")
cat("Anderson-Darling Test for AbstConc:\n", ad_abstconc, "\n\n")

# 3. Check linearity visually with a scatter plot and regression line
ggplot(AAIV_UNIQUE, aes(x = IMG, y = AbstConc)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_smooth(method = "lm", se = TRUE, color = "red") +
  labs(title = "Linearity Check: IMG vs AbstConc", x = "IMG", y = "AbstConc") +
  theme_minimal()

# COMPUTE CORRELATION BETWEEN IMG AND AbstConc
correlation_result <- cor.test(AAIV_UNIQUE$IMG, AAIV_UNIQUE$AbstConc, method = "pearson")
print(correlation_result)

################################################################################
# Check assumptions for linear regression
# To perform linear regression, we need to check the assumptions of linearity, homoscedasticity, and normality of residuals.

# 1. Linearity
# Test the correlation between word length (NCHAR) and abstraction (AbstConc) to assess the linearity of their relationship.
linearity_test <- cor.test(AAIV_clean$NCHAR, AAIV_clean$AbstConc)
cat("Correlation between word length and abstraction:\n")
print(linearity_test)

# 2. Homoscedasticity and Residual Analysis
# Fit a linear model to the data (AbstConc ~ NCHAR) to check for homoscedasticity (constant variance of residuals).
model <- lm(AbstConc ~ NCHAR, data = AAIV_clean)

# Plot residuals
# Plot the residuals from the linear model to visually inspect for homoscedasticity (constant variance of residuals).
par(mfrow = c(2, 2))  # Set up plotting layout for multiple plots
plot(model)

# Perform the Breusch-Pagan test for homoscedasticity
# The Breusch-Pagan test checks whether the variance of residuals is constant (homoscedasticity).
# A significant result (p-value < 0.05) would suggest heteroscedasticity, indicating varying variance in residuals.
install.packages("lmtest")
library(lmtest)
bp_test <- bptest(model)
cat("Breusch-Pagan Test for Homoscedasticity:\n")
print(bp_test)

# 3. Normality of Residuals
# Apply the Anderson-Darling test to check for normality
ad_test <- ad.test(residuals(model))
cat("Anderson-Darling Test for Normality of Residuals:\n")
print(ad_test)

# Given the significant violations of the assumptions of homoscedasticity and normality, 
# a standard linear regression is not appropriate. Instead, we use robust standard errors to adjust for heteroscedasticity 
# and ensure valid statistical inference.

library(sandwich)

# Compute robust standard errors for the model
robust_se <- coeftest(model, vcov = vcovHC(model, type = "HC3"))

# Display results with robust standard errors
cat("Linear Model with Robust Standard Errors:\n")
print(robust_se)



###############################################################################
# This section compares the relationship between word length and concreteness ratings using an English dataset (Brysbaert et al., 2013).
# The goal is to see whether a similar pattern to the German data is observed. Additionally, it examines whether compound nouns 
# (which are easily identified in English by hyphens or tabs) reduce the strength of the association between word length and abstraction,
# as we hypothesize that compound nouns in German exhibit this pattern.

# COMPARE THIS TO ENGLISH RATINGS (Brysbaert et al., 2013)

# Import dataset
# The dataset from Brysbaert et al. (2013) contains word concreteness ratings for English words.
Brysbaert <- data.table::fread("C:/Users/TinnerF/Dropbox/RCode/Aging/13428_2013_403_MOESM1_ESM.csv")

# Calculate Word_Length for all words
# The length of each word is calculated by removing spaces, hyphens, and tabs and counting the number of characters.
# This will help analyze if word length correlates with concreteness ratings in the English dataset.
Brysbaert$Word_Length <- sapply(Brysbaert$Word, function(x) {
  if (!is.na(x)) {
    nchar(gsub("\\s|-|\\t", "", x)) # Remove spaces, hyphens, and tabs before counting characters
  } else {
    NA
  }
})

# Check descriptive statistics
# Compute and display summary statistics for the concreteness ratings ('Conc.M') to understand the distribution.
library(psych)
describe(Brysbaert$Conc.M)

# Check for missing data
# Check if there are any missing values in the 'Word_Length' or 'Conc.M' columns.
# A warning is displayed if missing values are detected, and the data is cleaned by removing rows with missing values.
if (any(is.na(Brysbaert$Word_Length)) || any(is.na(Brysbaert$Conc.M))) {
  cat("Warning: Missing values detected. Proceeding after removing missing data.\n")
}

# Remove missing values
# Rows with missing 'Word_Length' or 'Conc.M' values are removed to ensure that the analysis is based on complete data.
Brysbaert_clean <- na.omit(Brysbaert[, .(Word_Length, Conc.M)])

# Plot the data to visualize linearity
# Create a scatter plot to visualize the relationship between word length and concreteness ratings.
# The plot includes a linear regression line to help assess the linearity of the relationship.
ggplot(Brysbaert_clean, aes(x = Word_Length, y = Conc.M)) +
  geom_point(alpha = 0.2) +
  geom_smooth(method = "lm", col = "blue") +
  labs(title = "Relationship Between Word Length and Concreteness",
       x = "Word Length (Number of Characters)",
       y = "Concreteness Rating") +
  theme_minimal()
# The plot shows a negative linear relationship between word length and concreteness ratings, with longer words tending to have lower concreteness ratings.

# Check assumptions for linear regression
# 1. Linearity
# Check the correlation between word length and concreteness to test for linearity of the relationship.
linearity_test <- cor.test(Brysbaert_clean$Word_Length, Brysbaert_clean$Conc.M)
cat("Correlation between word length and concreteness:\n")
print(linearity_test)
# The Pearson correlation test indicates a moderate negative correlation between word length and concreteness ratings (r = -0.32, p-value < 2.2e-16), suggesting that as word length increases, concreteness ratings tend to decrease.

# 2. Homoscedasticity and Residual Analysis
# Fit a linear model to the data (Concreteness ~ Word_Length) and check for homoscedasticity (constant variance of residuals).
model <- lm(Conc.M ~ Word_Length, data = Brysbaert_clean)

# Plot residuals
# Create residual plots to visually check for homoscedasticity (constant variance).
par(mfrow = c(2, 2))  # Set up a plotting layout for multiple plots
plot(model)
# The residual plots suggest potential issues with homoscedasticity, as the "Residuals vs Fitted" and "Scale-Location" plots 
# show a non-constant variance, indicating that the assumption of equal variance of residuals may not be fully met.

# Apply log transformation to 'Conc.M' to address potential heteroscedasticity
# This transformation may help stabilize the variance and improve homoscedasticity.
model_transformed <- lm(log(Conc.M) ~ Word_Length, data = Brysbaert_clean)

# Plot residuals again after transformation
par(mfrow = c(2, 2))
plot(model_transformed)

# Perform the Breusch-Pagan test for homoscedasticity on the transformed model
library(lmtest)
bp_test <- bptest(model_transformed)
cat("Breusch-Pagan Test for Homoscedasticity (Transformed Model):\n")
print(bp_test)
# The Breusch-Pagan test for the transformed model indicates significant evidence of heteroscedasticity (p-value < 2.2e-16),
# suggesting that the variance of residuals is still not constant, even after the transformation.

# Install the nortest package if not already installed
# install.packages("nortest")
library(nortest)

# Apply the Anderson-Darling test to check for normality
ad_test <- ad.test(residuals(model_transformed))
cat("Anderson-Darling Test for Normality of Residuals:\n")
print(ad_test)
# The Anderson-Darling test indicates that the residuals significantly deviate from a normal distribution (p-value < 2.2e-16), confirming that the normality assumption is violated.

# Based on the outputs, the next step should be to proceed with the original linear regression model 
# but compute robust standard errors to account for the significant violations of homoscedasticity (Breusch-Pagan test, p-value < 2.2e-16)
# and normality (Anderson-Darling test, p-value < 2.2e-16), ensuring reliable statistical inference.

# The 'sandwich' package:
# Provides tools for computing robust covariance matrix estimators for regression models.
# These estimators adjust standard errors to account for violations of assumptions such as heteroscedasticity
# or autocorrelation, making statistical inference more reliable.
library(sandwich)

# The 'lmtest' package:
# Provides a suite of diagnostic tools and hypothesis tests for linear models.
# It includes tests such as the Breusch-Pagan test for heteroscedasticity and the Durbin-Watson test for autocorrelation.
# Often used in conjunction with the 'sandwich' package to test models with robust standard errors.
library(lmtest)

# Compute robust standard errors for the original model
robust_se <- coeftest(model, vcov = vcovHC(model, type = "HC3"))

# View the results with robust standard errors
cat("Linear Model with Robust Standard Errors:\n")
print(robust_se)

# There is a strong and statistically significant negative relationship between 
# word length and concreteness ratings. Using robust standard errors ensures that 
# the results are reliable even in the presence of assumption violations, particularly heteroscedasticity.


# COMPARE RATINGS FOR COMPOUND VERSUS NON-COMPOUND WORDS - WHICH ARE EASY TO IDENTIFY USING REGEX
# In this part, we explore whether compound words (e.g., "mother-in-law") show a different pattern in the relationship
# between word length and concreteness. Compound words are identified using a regular expression (hyphen or tab).
# This analysis helps assess whether compound words reduce the association between word length and abstraction/concreteness.

# Add a variable identifying compound words (hyphen or tab between words)
Brysbaert$Compound <- ifelse(grepl("-|\\t", Brysbaert$Word), 1, 0)

# Calculate the mean concreteness rating for compound and non-compound words
# Compute the mean concreteness rating for compound and non-compound words to see if there's a difference in concreteness ratings.
mean_concreteness <- aggregate(Conc.M ~ Compound, data = Brysbaert, mean)

print(mean_concreteness)

# Ensure numeric types for correlation analysis
# Convert 'Word_Length' and 'Conc.M' columns to numeric to ensure they are properly formatted for correlation analysis.
Brysbaert$Word_Length <- as.numeric(Brysbaert$Word_Length)
Brysbaert$Conc.M <- as.numeric(Brysbaert$Conc.M)


# Check for NA values in Word_Length and Conc.M for compound words
compound_valid <- !is.na(Brysbaert$Word_Length[Brysbaert$Compound == 1]) & 
  !is.na(Brysbaert$Conc.M[Brysbaert$Compound == 1])

if (sum(compound_valid) > 1) {
  # Compute correlation if there are sufficient valid pairs for compound words
  compound_correlation <- cor(
    Brysbaert$Word_Length[Brysbaert$Compound == 1][compound_valid], 
    Brysbaert$Conc.M[Brysbaert$Compound == 1][compound_valid]
  )
} else {
  compound_correlation <- NA
  cat("Not enough valid data for compound words.\n")
}

# Check for NA values in Word_Length and Conc.M for non-compound words
non_compound_valid <- !is.na(Brysbaert$Word_Length[Brysbaert$Compound == 0]) & 
  !is.na(Brysbaert$Conc.M[Brysbaert$Compound == 0])

if (sum(non_compound_valid) > 1) {
  # Compute Spearman's correlation if there are sufficient valid pairs for non-compound words
  non_compound_correlation <- cor.test(
    Brysbaert$Word_Length[Brysbaert$Compound == 0][non_compound_valid], 
    Brysbaert$Conc.M[Brysbaert$Compound == 0][non_compound_valid], 
    method = "spearman"
  )
  cat("Spearman's Correlation for Non-Compound Words:\n")
  print(non_compound_correlation)
} else {
  non_compound_correlation <- NA
  cat("Not enough valid data for non-compound words.\n")
}

# Print the results
cat("Correlation between Word Length and Concreteness:\n")
cat("Compound Words: ", compound_correlation, "\n")

# Check if non_compound_correlation is valid and extract values
if (!is.null(non_compound_correlation) && !is.na(non_compound_correlation$estimate)) {
  cat("Non-Compound Words: rho =", non_compound_correlation$estimate, 
      ", p-value =", non_compound_correlation$p.value, "\n")
} else {
  cat("Non-Compound Words: Not enough valid data.\n")
}

################################################################################

# Rename Variable For Merging
POS_UNIQUE <- POS_UNIQUE %>% rename(Word = lemma_cleaned)
# MERGE DATA SETS 
# Perform the inner join while avoiding .x and .y duplicates

# Perform the inner join while retaining and resolving duplicates
WordType <- inner_join(POS_UNIQUE, AAIV_UNIQUE, by = "Word") %>%
  # Select and rename duplicate columns
  mutate(NCHAR = coalesce(NCHAR.x, NCHAR.y)) %>%  # Merge NCHAR.x and NCHAR.y into a single column
  select(-NCHAR.x, -NCHAR.y)  # Remove redundant columns

#################################################################################

# SCATTERPLOT - WORD LENGTH VS ABSTRACTION RATING

# Check Assumptions For Slope Calculation

# Linearity Check
ggplot(WordType, aes(x = NCHAR, y = AbstConc)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", color = "blue") +
  labs(title = "Linearity Check: NCHAR vs AbstConc",
       x = "Word Length (NCHAR)",
       y = "Abstractness (AbstConc)") +
  theme_minimal()

# 2.2 Fit Model to Calculate Residuals
model_abst <- lm(AbstConc ~ NCHAR, data = WordType)

# 2.3 Check Homoscedasticity
par(mfrow = c(2, 2))  # Diagnostic plots
plot(model_abst)       # Check residuals visually

# Breusch-Pagan Test for Homoscedasticity
bp_test <- bptest(model_abst)
cat("Breusch-Pagan Test for Homoscedasticity:\n")
print(bp_test)

# 2.4 Check Normality of Residuals
# Q-Q Plot for Residuals
qqnorm(residuals(model_abst))
qqline(residuals(model_abst), col = "red", lwd = 2)

# Anderson-Darling Test for Normality
ad_test <- ad.test(residuals(model_abst))
cat("Anderson-Darling Test for Normality of Residuals:\n")
print(ad_test)

# Step 3: Run the Regression if Assumptions Are Met
if (bp_test$p.value > 0.05 & ad_test$p.value > 0.05) {
  slope_abst <- coef(model_abst)[2]  # Extract the slope
  cat("Slope of the relationship between NCHAR and AbstConc:\n", slope_abst, "\n")
} else {
  cat("Regression assumptions are violated. Consider using robust methods or transformations.\n")
}

# Fit the linear model
model_abst <- lm(AbstConc ~ NCHAR, data = WordType)

# Calculate robust standard errors and extract slope
robust_se <- coeftest(model_abst, vcov = vcovHC(model_abst, type = "HC3"))
robust_slope <- robust_se["NCHAR", "Estimate"]  # Extract the slope for robust model
print(robust_se)
print(robust_slope)

WordTypeUpos <- WordType %>% group_by(upos, NCHAR) %>%
  summarize(
  AbstConc = mean(AbstConc),
  N = n())

AbstConc <- ggplot(WordTypeUpos, aes(x = NCHAR, y = AbstConc, size = N, color = upos)) +
  geom_point(alpha = 0.6) +  # Plot points with transparency
  geom_abline(
    intercept = coef(model_abst)[1],  # Intercept from the robust model
    slope = robust_slope,            # Slope from the robust model
    color = "red", size = 1, linetype = "dashed"  # Dashed red line for robust model
  ) +
  labs(
    x = "Word Length",
    y = "Concreteness",
    color = "Lexical Class",
    size = "Number of Words"
  ) +
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.line = element_line(colour = "black"),
    text = element_text(size = 20),
    axis.text.x = element_text(angle = 0, size = 20),
    axis.text.y = element_text(size = 20),
    legend.direction = "horizontal",
    legend.position = "bottom",
    legend.text = element_text(size = 16),  # Slightly reduced legend text size for alignment
    legend.title = element_text(size = 16)  # Slightly reduced legend title size for alignment
  ) +
  scale_size_continuous(range = c(4, 12)) +  # Adjust size range for larger circles
  guides(
    color = guide_legend(
      title.position = "top",
      override.aes = list(size = 6),
      order = 1  # Ensure 'color' legend appears first
    ),
    size = guide_legend(
      title.position = "top",
      order = 2  # Ensure 'size' legend appears second
    )
  ) +
  ylim(2, 5.5) +  # Crop the y-axis range to remove empty space
  # Add the slope from the robust model as an annotation
  annotate(
    "text",
    x = 1,  # Adjust the x position for better placement
    y = 2.1,  # Position near the top of the y-axis range
    label = paste("Robust Slope =", round(robust_slope, 3)),  # Add slope and round it to 3 decimal places
    size = 7,  # Adjust text size
    hjust = 0
  )

# Print the plot
print(AbstConc)


################################################################################
# Compute Slope for Nouns only
GroupByNOUNS <- WordType %>%
  filter(upos == "NOUN") %>%  # Filter for upos == "Noun"
  group_by(upos, NCHAR) %>%
  summarize(
    AbstConc = mean(AbstConc, na.rm = TRUE),
    IMG = mean(IMG, na.rm = TRUE),
    N = n(),
    .groups = "drop"  # Avoid grouping further after summarization
  )

# Fit linear model
model_abst <- lm(AbstConc ~ NCHAR, data = GroupByNOUNS)

# Compute robust standard errors
robust_se <- coeftest(model_abst, vcov = vcovHC(model_abst, type = "HC1"))

# Extract slope and robust standard error
slope_abstNOUNS <- robust_se[2, "Estimate"]  # Slope
robust_slope_se <- robust_se[2, "Std. Error"]  # Robust standard error

print(slope_abstNOUNS)
print(robust_slope_se)

# RECALCULATE SLOPE FOR NOUNS ONLY AGAIN TO SEE WHETHER THE ASSOCIATION BETWEEN 
# Filter the data to include only words with fewer than 25 characters
FilteredWordType <- WordType %>%
  filter(NCHAR <= 25)  # Keep only words with fewer than 25 characters

# Compute Slope for Nouns only from the filtered dataset
GroupByNOUNS <- FilteredWordType %>%
  filter(upos == "NOUN") %>%  # Filter for upos == "Noun"
  group_by(upos, NCHAR) %>%
  summarize(
    AbstConc = mean(AbstConc, na.rm = TRUE),
    IMG = mean(IMG, na.rm = TRUE),
    N = n(),
    .groups = "drop"  # Avoid grouping further after summarization
  )

# Fit linear model
model_abst <- lm(AbstConc ~ NCHAR, data = GroupByNOUNS)

# Compute robust standard errors
robust_se <- coeftest(model_abst, vcov = vcovHC(model_abst, type = "HC1"))

# Extract slope and robust standard error
slope_abstNOUNS <- robust_se[2, "Estimate"]  # Slope
robust_slope_se <- robust_se[2, "Std. Error"]  # Robust standard error

# Print results
cat("Robust Slope for NCHAR and AbstConc (NOUNs):\n", slope_abstNOUNS, "\n")
cat("Robust Standard Error for Slope:\n", robust_slope_se, "\n")

###############################################################################
# SCATTERPLOT - WORD LENGTH VS IMAGINEABILITY

# Step 1: Linearity Check
ggplot(WordType, aes(x = NCHAR, y = IMG)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", color = "blue") +
  labs(
    title = "Linearity Check: NCHAR vs IMG",
    x = "Word Length (NCHAR)",
    y = "Imagineability (IMG)"
  ) +
  theme_minimal()

# Step 2: Fit the Model and Assumption Checks

# 2.1 Fit Linear Model
model_img <- lm(IMG ~ NCHAR, data = WordType)

# 2.2 Homoscedasticity Check
par(mfrow = c(2, 2))  # Diagnostic plots for residuals
plot(model_img)        # Visual inspection of residuals

# Breusch-Pagan Test
bp_test <- bptest(model_img)
cat("Breusch-Pagan Test for Homoscedasticity:\n")
print(bp_test)

# 2.3 Normality Check of Residuals
# Q-Q Plot
qqnorm(residuals(model_img))
qqline(residuals(model_img), col = "red", lwd = 2)

# Anderson-Darling Test
ad_test <- ad.test(residuals(model_img))
cat("Anderson-Darling Test for Normality of Residuals:\n")
print(ad_test)

# Step 3: Conditional Regression and Slope Calculation

if (bp_test$p.value > 0.05 & ad_test$p.value > 0.05) {
  slope_img <- coef(model_img)[2]  # Extract slope
  cat("Slope of the relationship between NCHAR and IMG:\n", slope_img, "\n")
} else {
  cat("Regression assumptions are violated. Using robust methods.\n")
  
  # Calculate robust slope
  robust_se <- coeftest(model_img, vcov = vcovHC(model_img, type = "HC3"))
  robust_slope <- robust_se["NCHAR", "Estimate"]
  print(robust_se)
  print(robust_slope)
}

# Aggregate Data
GroupByUpos <- WordType %>%
  group_by(upos, NCHAR) %>%
  summarize(
    IMG = mean(IMG),
    N = n(),
    .groups = "drop"  # Avoid grouping issues
  )

# Visualization
Imagineability <- ggplot(GroupByUpos, aes(x = NCHAR, y = IMG, size = N, color = upos)) +
  geom_point(alpha = 0.6) +
  geom_abline(
    intercept = coef(model_img)[1],
    slope = ifelse(exists("robust_slope"), robust_slope, slope_img),
    color = "red", size = 1, linetype = "dashed"
  ) +
  labs(
    x = "Word Length",
    y = "Imagineability",
    color = "Lexical Class",
    size = "Number of Words"
  ) +
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.line = element_line(colour = "black"),
    text = element_text(size = 20),
    axis.text.x = element_text(angle = 0, size = 20),
    axis.text.y = element_text(size = 20),
    legend.direction = "horizontal",
    legend.position = "bottom",
    legend.text = element_text(size = 16),
    legend.title = element_text(size = 16)
  ) +
  scale_size_continuous(range = c(3, 12)) +  # Adjusted size range for better visualization
  guides(
    color = guide_legend(title.position = "top", override.aes = list(size = 6), order = 1),
    size = guide_legend(title.position = "top", order = 2)
  ) +
  ylim(3, 6) +
  annotate(
    "text",
    x = 1,
    y = 3.1,
    label = paste("Robust Slope =", round(ifelse(exists("robust_slope"), robust_slope, slope_img), 3)),
    size = 7,
    hjust = 0
  )

# Print the updated plot
print(Imagineability)



###############################################################################
# Combine both plots into one layout with a common legend
combined_plot <- ggpubr::ggarrange(
  AbstConc, Imagineability, 
  ncol = 2, 
  nrow = 1, 
  labels = c("A", "B"), 
  common.legend = TRUE, 
  legend = "bottom"
)

print(combined_plot)
################################################################################
################################################################################
# AGE ANALYSIS
#################################################################################
#################################################################################
# This section examines how WordCount and WordLength vary across the lifespan.
# Since longitudinal observations are unavailable, reviews are aggregated by 
# reviewer age to derive trajectories of these linguistic variables over the lifespan.
################################################################################
# Import Reviews

# Set Workspace
setwd("C:/Users/TinnerF/Dropbox/RCode/Aging")

# Import Customer Data (Age & Reviews)
df_consumer <- data.table::fread("df_consumer.csv", encoding = "UTF-8") 
df_consumer$Age <- round(df_consumer$kundenalter)
df_consumer$V1 <- NULL

################################################################################ 
# Age Distribution
# Note:
# The 'kundenalter' column contains fractional age values because it was computed 
# from the date of birth using the current machine date. These fractional values 
# represent the exact age in years (e.g., 40.5 for 40 years and 6 months). 
# To ensure a clean and interpretable histogram with consistent binning, we apply 
# floor rounding to convert ages to whole numbers, effectively representing the 
# last completed year of age.
df_consumer$ReviewerAge <- floor(df_consumer$kundenalter)

# Ensure There Are No Missing Values (Leaves Us With A Total of 375748 Observations)
df_consumer <- df_consumer %>% filter(!is.na(ReviewerAge))

# Create the plot
ggplot(df_consumer, aes(x = ReviewerAge)) +
  geom_histogram(binwidth = 1, fill = "steelblue", color = "black", alpha = 0.7) +
  labs(
    title = "",
    x = "Age",
    y = "Frequency"
  ) +
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.line = element_line(colour = "black"),
    text = element_text(size = 20),
    axis.text.x = element_text(angle = 0, size = 20),
    axis.text.y = element_text(size = 20),
    legend.direction = "horizontal",
    legend.position = "bottom",
    legend.text = element_text(size = 20),
    legend.title = element_text(size = 20)
  )

library(ggplot2)
library(dplyr)

# Ensure there are no missing values for ReviewerAge and anredeID
df_filtered <- df_consumer %>%
  filter(!is.na(ReviewerAge) & !is.na(anredeID) & anredeID %in% c(1, 2))

# Create the stacked bar chart with customized colors and labels
ggplot(df_filtered, aes(x = ReviewerAge, fill = as.factor(anredeID))) +
  geom_bar(color = "black", alpha = 0.7, position = "stack") +
  scale_fill_manual(
    values = c("1" = "cornflowerblue", "2" = "firebrick2"),  # Assign the same color to both groups
    labels = c("1" = "male", "2" = "female")           # Customize legend labels
  ) +
  labs(
    title = "",
    x = "Age",
    y = "Frequency",
    fill = "Gender"
  ) +
  theme_minimal() +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    axis.line = element_line(colour = "black"),
    text = element_text(size = 20),
    axis.text.x = element_text(angle = 0, size = 20),
    axis.text.y = element_text(size = 20),
    legend.direction = "horizontal",
    legend.position = "bottom",
    legend.text = element_text(size = 20),
    legend.title = element_text(size = 20)
  )
library(psych)
describe(df_consumer$Age)
################################################################################
# DEFINE VARIABLE THAT CAPTURES LEVEL OF ABSTRACTION

# Step 1: Calculate the average word length per review (first clean text properly)
library(stringr)

df_consumer <- df_consumer %>%
  mutate(
    # Clean the text
    cleaned_text = text %>%
      # Remove repetitive patterns like "usususususus..."
      str_replace_all("([a-zA-ZäöüÄÖÜß])\\1{4,}", "\\1") %>%
      
      # Replace '+' with whitespace for cases like "Volleyball+Dartspielen+..."
      str_replace_all("\\+", " ") %>%
      
      # Remove non-alphabetic characters (excluding German letters) and replace with whitespace
      str_replace_all("[^a-zA-ZäöüÄÖÜß\\s]", " ") %>%
      
      # Split compound words before internal capitalization (e.g., "HotelSchöner")
      str_replace_all("([a-zäöüß])([A-ZÄÖÜ])", "\\1 \\2") %>%
      
      # Replace repeated whitespace with a single space
      str_replace_all("\\s{2,}", " ") %>%
      
      # Trim leading and trailing whitespace
      str_trim() %>%
      
      # Remove words with repeated characters or gibberish (e.g., "bhjkhjkjk")
      str_replace_all("\\b[a-zA-ZäöüÄÖÜß]{2,}\\b(?=.*(\\w{2}\\1))", "") %>%
      
      # Remove text containing a high ratio of consonants
      str_replace_all("\\b[^aeiouäöüAEIOUÄÖÜ\\s]{5,}\\b", ""),
    
    # Split cleaned text into words and calculate word lengths
    word_lengths = str_split(cleaned_text, "\\s+") %>%
      lapply(function(words) nchar(words[words != ""])), # Filter out empty strings
    
    # Calculate average word length per review
    WordLength = sapply(word_lengths, function(lengths) if(length(lengths) > 0) mean(lengths) else NA)
  ) %>%
  select(-word_lengths)  # Remove intermediate column


# CHEK IF TEXT HAS BEEN ACCORDINGLY CLEANED
# IDENTIFY EXTREMELY LARGE WORDS (that might indicate gibberish)

# Identify rows with words exceeding length of 40
outliers <- df_consumer %>%
  rowwise() %>%
  mutate(has_long_word = any(str_length(unlist(str_split(cleaned_text, "\\s+"))) > 20)) %>%
  filter(has_long_word) %>%
  select(cleaned_text)

# View outliers
print(outliers)

###############################################################################
# Descriptive Statistics

install.packages("psych")
library(psych)
describe(df_consumer$WordCount)
describe(df_consumer$WordLength)
describe(df_consumer$PositiveEmotions)
describe(df_consumer$NegativeEmotions)
describe(df_consumer$ReviewerRating)


# Due to WordCount being heavily skewed and both WordCount and WordLength exhibiting a heavy-tailed 
# distribution, we log-transform the data
###############################################################################
# CORRELATOINS

# Create log-transformed variables for WordCount and WordLength
df_consumer <- df_consumer %>%
  mutate(
    LogWordCount = log(WordCount + 1),
    LogWordLength = log(WordLength + 1),  
    LogPositiveEmotions = log(PositiveEmotions + 1),
    LogNegativeEmotions = log(NegativeEmotions + 1),
    LogDaysPassedReturnVacation = log(DaysPassedReturnVacation + 1),
    LogVacationDuration = log(VacationDuration + 1)
  )

# Spearman correlation (no transformation)
cor.test(df_consumer$WordCount, df_consumer$WordLength, method = "spearman")
cor.test(df_consumer$WordCount, df_consumer$NegativeEmotions, method = "spearman")
cor.test(df_consumer$WordLength, df_consumer$PositiveEmotions, method = "spearman")


# CORRELATION (GROUPED)
grouped <- df_consumer %>%
  group_by(ReviewerAge) %>%
  summarize(
    WordCount = mean(WordCount, na.rm = TRUE),
    WordLength = mean(WordLength, na.rm = TRUE),
    NegativeEmotions = mean(NegativeEmotions, na.rm = TRUE),
    PositiveEmotions = mean(PositiveEmotions, na.rm = TRUE)
  )

# Spearman correlation (no transformation)
cor.test(grouped$WordCount, grouped$WordLength, method = "spearman")
cor.test(grouped$WordCount, grouped$NegativeEmotions, method = "spearman")
cor.test(grouped$WordLength, grouped$PositiveEmotions, method = "spearman")


################################################################################
# Ensure required libraries are loaded
library(lmtest)
library(sandwich)

# 1) Prepare Data: Remove rows with missing values
df_consumer <- df_consumer[!is.na(df_consumer$LogWordLength) & !is.na(df_consumer$LogWordCount), ]

# 2) Simple Linear Regression for WordCount and WordLength
model_word_count <- lm(LogWordCount ~ ReviewerAge, data = df_consumer)
model_word_length <- lm(LogWordLength ~ ReviewerAge, data = df_consumer)

# 3) TEST ASSUMPTIONS FOR BOTH MODELS

# LINEARITY
cat("\nLinearity: Residuals vs. Fitted for WordCount:\n")
plot(model_word_count, which = 1)  # WordCount
cat("\nLinearity: Residuals vs. Fitted for WordLength:\n")
plot(model_word_length, which = 1)  # WordLength

# HOMOSCEDASTICITY: Breusch-Pagan Test
cat("\nHomoscedasticity (Breusch-Pagan Test):\n")
cat("WordCount: ")
print(bptest(model_word_count))  # WordCount
cat("WordLength: ")
print(bptest(model_word_length))  # WordLength

# NORMALITY OF RESIDUALS: Q-Q Plot and Anderson-Darling Test
cat("\nNormality of Residuals:\n")
cat("Q-Q Plot for WordCount:\n")
plot(model_word_count, which = 2)  # Q-Q plot for WordCount
cat("Q-Q Plot for WordLength:\n")
plot(model_word_length, which = 2)  # Q-Q plot for WordLength

# Anderson-Darling Test
library(nortest)
cat("Anderson-Darling Test for WordCount:\n")
print(ad.test(residuals(model_word_count)))
cat("Anderson-Darling Test for WordLength:\n")
print(ad.test(residuals(model_word_length)))

# INDEPENDENCE OF RESIDUALS: Durbin-Watson Test
cat("\nIndependence of Residuals (Durbin-Watson Test):\n")
cat("WordCount: ")
print(dwtest(model_word_count))
cat("WordLength: ")
print(dwtest(model_word_length))

# 4) COMPUTE ROBUST STANDARD ERRORS
cat("\nRobust Standard Errors:\n")
# WordCount
robust_se_word_count <- coeftest(model_word_count, vcov = vcovHC(model_word_count, type = "HC3"))
cat("WordCount:\n")
print(robust_se_word_count)

# WordLength
robust_se_word_length <- coeftest(model_word_length, vcov = vcovHC(model_word_length, type = "HC3"))
cat("WordLength:\n")
print(robust_se_word_length)

# 5) COMPUTE CHANGES FROM ROBUST STANDARD ERRORS
compute_changes_robust <- function(robust_se, data, dependent_var) {
  # Extract the slope
  slope <- robust_se["ReviewerAge", "Estimate"]
  
  # Back-transform the slope to the original scale
  percent_change_per_unit <- (exp(slope) - 1) * 100
  
  # Compute the mean of the dependent variable in the original scale
  mean_dep_var <- mean(data[[dependent_var]], na.rm = TRUE)
  
  # Compute the absolute change in the original scale
  absolute_change_per_unit <- mean_dep_var * (1 - exp(slope))
  
  # Return results
  list(
    slope = slope,
    percent_change_per_unit = percent_change_per_unit,
    absolute_change_per_unit = absolute_change_per_unit
  )
}

# Apply for WordCount
cat("\nChanges from Robust SE for WordCount:\n")
result_word_count_robust <- compute_changes_robust(robust_se_word_count, df_consumer, "WordCount")
cat(sprintf("Slope (log-transformed scale): %.5f\n", result_word_count_robust$slope))
cat(sprintf("Percent change per unit (age): %.2f%%\n", result_word_count_robust$percent_change_per_unit))
cat(sprintf("Absolute change per unit (age): %.2f words\n", result_word_count_robust$absolute_change_per_unit))

# Apply for WordLength
cat("\nChanges from Robust SE for WordLength:\n")
result_word_length_robust <- compute_changes_robust(robust_se_word_length, df_consumer, "WordLength")
cat(sprintf("Slope (log-transformed scale): %.5f\n", result_word_length_robust$slope))
cat(sprintf("Percent change per unit (age): %.4f%%\n", result_word_length_robust$percent_change_per_unit))
cat(sprintf("Absolute change per unit (age): %.4f characters\n", result_word_length_robust$absolute_change_per_unit))


#################################################################################
# Check if Quadratic or Logarithmic Term for ReviewerAge Improves Models for WordCount and WordLength

# 1. Add quadratic term for ReviewerAge for WordCount
model_word_count_quad <- lm(LogWordCount ~ ReviewerAge + I(ReviewerAge^2), data = df_consumer)
summary(model_word_count_quad)

# Compute robust standard errors for quadratic model (WordCount)
install.packages("lmtest")
library(lmtest)
robust_se_word_count_quad <- coeftest(model_word_count_quad, vcov = vcovHC(model_word_count_quad, type = "HC3"))
print(robust_se)

cat("\nRobust Standard Errors for Quadratic WordCount Model:\n")
print(robust_se_word_count_quad)

# Add quadratic term for ReviewerAge for WordLength
model_word_length_quad <- lm(LogWordLength ~ ReviewerAge + I(ReviewerAge^2), data = df_consumer)
summary(model_word_length_quad)

# Compute robust standard errors for quadratic model (WordLength)
robust_se_word_length_quad <- coeftest(model_word_length_quad, vcov = vcovHC(model_word_length_quad, type = "HC3"))
cat("\nRobust Standard Errors for Quadratic WordLength Model:\n")
print(robust_se_word_length_quad)

# 2. Log-transform ReviewerAge for WordCount
df_consumer$LogReviewerAge <- log(df_consumer$ReviewerAge)
model_word_count_log_age <- lm(LogWordCount ~ LogReviewerAge, data = df_consumer)
summary(model_word_count_log_age)

# Compute robust standard errors for log-transformed model (WordCount)
robust_se_word_count_log_age <- coeftest(model_word_count_log_age, vcov = vcovHC(model_word_count_log_age, type = "HC3"))
cat("\nRobust Standard Errors for Log-Transformed ReviewerAge WordCount Model:\n")
print(robust_se_word_count_log_age)

# Log-transform ReviewerAge for WordLength
model_word_length_log_age <- lm(LogWordLength ~ LogReviewerAge, data = df_consumer)
summary(model_word_length_log_age)

# Compute robust standard errors for log-transformed model (WordLength)
robust_se_word_length_log_age <- coeftest(model_word_length_log_age, vcov = vcovHC(model_word_length_log_age, type = "HC3"))
cat("\nRobust Standard Errors for Log-Transformed ReviewerAge WordLength Model:\n")
print(robust_se_word_length_log_age)

# 3. Compare models using AIC/BIC for WordCount
cat("\nModel Comparisons for WordCount (AIC and BIC):\n")
aic_values_word_count <- AIC(model_word_count, model_word_count_quad, model_word_count_log_age)
bic_values_word_count <- BIC(model_word_count, model_word_count_quad, model_word_count_log_age)
cat("\nAIC Values (WordCount):\n")
print(aic_values_word_count)
cat("\nBIC Values (WordCount):\n")
print(bic_values_word_count)

# 3. Compare models using AIC/BIC for WordLength
cat("\nModel Comparisons for WordLength (AIC and BIC):\n")
aic_values_word_length <- AIC(model_word_length, model_word_length_quad, model_word_length_log_age)
bic_values_word_length <- BIC(model_word_length, model_word_length_quad, model_word_length_log_age)
cat("\nAIC Values (WordLength):\n")
print(aic_values_word_length)
cat("\nBIC Values (WordLength):\n")
print(bic_values_word_length)

# Conclusion: Minor Improvement in AIC for the Quadratic Model - Improvement however is so minimal that we stick with the linera model 
# for ease of interpretability
########################################################################################
# ADD COVARIATES

# Define a function to summarize linear models with robust SE
summarize_linear_model <- function(model, dv, covariates, data) {
  # Compute robust standard errors using sandwich package
  robust_se <- coeftest(model, vcov = vcovHC(model, type = "HC3"))
  
  # Compute standardized beta coefficients
  beta_standardized <- coef(model)[covariates] * sd(data[[covariates]], na.rm = TRUE) / sd(data[[dv]], na.rm = TRUE)
  
  # Extract R-squared, residual standard error, and AIC
  r_squared <- summary(model)$r.squared
  residual_se <- summary(model)$sigma
  aic <- AIC(model)
  
  list(
    robust_se = robust_se,
    beta_standardized = beta_standardized,
    r_squared = r_squared,
    residual_se = residual_se,
    aic = aic
  )
}

# Fit linear model for WordCount
model_word_count <- lm(
  LogWordCount ~ ReviewerAge + LogVacationDuration + LogDaysPassedReturnVacation + ReviewerRating,
  data = na.omit(df_consumer)
)
result_word_count <- summarize_linear_model(model_word_count, "LogWordCount", "ReviewerAge", df_consumer)

# Fit linear model for WordLength
model_word_length <- lm(
  LogWordLength ~ ReviewerAge + LogVacationDuration + LogDaysPassedReturnVacation + ReviewerRating,
  data = na.omit(df_consumer)
)
result_word_length <- summarize_linear_model(model_word_length, "LogWordLength", "ReviewerAge", df_consumer)

# Back-transform results for interpretability
mean_word_count <- mean(df_consumer$WordCount, na.rm = TRUE)
mean_word_length <- mean(df_consumer$WordLength, na.rm = TRUE)

reduction_per_year_word_count <- mean_word_count * (exp(coef(model_word_count)["ReviewerAge"]) - 1)
reduction_10_years_word_count <- mean_word_count * (1 - exp(10 * coef(model_word_count)["ReviewerAge"]))

increase_per_year_word_length <- mean_word_length * (exp(coef(model_word_length)["ReviewerAge"]) - 1)
increase_10_years_word_length <- mean_word_length * ((1 + (exp(coef(model_word_length)["ReviewerAge"]) - 1))^10 - 1)

# Results summary
cat("WordCount Model:\n",
    "Each additional year of age was associated with an average reduction of", round(reduction_per_year_word_count, 2), "words per review,\n",
    "amounting to a cumulative decrease of approximately", round(reduction_10_years_word_count, 2), "words over a decade.\n",
    "Standardized beta:", round(result_word_count$beta_standardized, 4), "\n",
    "R²:", round(result_word_count$r_squared, 4), "\n",
    "Residual SE:", round(result_word_count$residual_se, 4), "\n",
    "AIC:", round(result_word_count$aic, 2), "\n\n")

cat("WordLength Model:\n",
    "Each additional year of age was associated with an average increase of", round(increase_per_year_word_length, 5), "characters per word,\n",
    "resulting in a cumulative increase of approximately", round(increase_10_years_word_length, 5), "characters over a decade.\n",
    "Standardized beta:", round(result_word_length$beta_standardized, 4), "\n",
    "R²:", round(result_word_length$r_squared, 4), "\n",
    "Residual SE:", round(result_word_length$residual_se, 4), "\n",
    "AIC:", round(result_word_length$aic, 2), "\n\n")

########################################################################################
# MIXED EFFECT MODELS (HID AS FIXED EFFECT)

# Load required packages
if (!require(lme4)) install.packages("lme4", dependencies = TRUE)
if (!require(MuMIn)) install.packages("MuMIn", dependencies = TRUE)
if (!require(clubSandwich)) install.packages("clubSandwich", dependencies = TRUE)
library(lme4)
library(MuMIn)
library(clubSandwich)

# Define a function to summarize mixed-effects models with robust SE
summarize_mixed_model <- function(model, dv, covariates, data) {
  # Extract the grouping variable from the model
  model_data <- model.frame(model)
  cluster_variable <- model_data$hid
  
  # Compute robust standard errors using clubSandwich
  robust_se <- coef_test(model, vcov = "CR2", cluster = cluster_variable)
  
  # Compute standardized beta coefficients
  beta_standardized <- fixef(model)[covariates] * sd(data[[covariates]], na.rm = TRUE) / sd(data[[dv]], na.rm = TRUE)
  
  # Compute marginal and conditional R-squared
  r_squared_marginal <- r.squaredGLMM(model)[1]
  r_squared_conditional <- r.squaredGLMM(model)[2]
  
  # Extract residual standard error and AIC
  residual_se <- sigma(model)
  aic <- AIC(model)
  
  list(
    robust_se = robust_se,
    beta_standardized = beta_standardized,
    r_squared_marginal = r_squared_marginal,
    r_squared_conditional = r_squared_conditional,
    residual_se = residual_se,
    aic = aic
  )
}

# Fit mixed-effects model for WordCount
model_word_count <- lmer(
  LogWordCount ~ ReviewerAge + LogVacationDuration + LogDaysPassedReturnVacation + ReviewerRating + (1 | hid),
  data = na.omit(df_consumer),
  REML = FALSE
)
result_word_count <- summarize_mixed_model(model_word_count, "LogWordCount", "ReviewerAge", df_consumer)

# Fit mixed-effects model for WordLength
model_word_length <- lmer(
  LogWordLength ~ ReviewerAge + LogVacationDuration + LogDaysPassedReturnVacation + ReviewerRating + (1 | hid),
  data = na.omit(df_consumer),
  REML = FALSE
)
result_word_length <- summarize_mixed_model(model_word_length, "LogWordLength", "ReviewerAge", df_consumer)

# Back-transform results for interpretability
mean_word_count <- mean(df_consumer$WordCount, na.rm = TRUE)
mean_word_length <- mean(df_consumer$WordLength, na.rm = TRUE)

reduction_per_year_word_count <- mean_word_count * (exp(fixef(model_word_count)["ReviewerAge"]) - 1)
reduction_10_years_word_count <- mean_word_count * (1 - exp(10 * fixef(model_word_count)["ReviewerAge"]))

increase_per_year_word_length <- mean_word_length * (exp(fixef(model_word_length)["ReviewerAge"]) - 1)
increase_10_years_word_length <- mean_word_length * ((1 + (exp(fixef(model_word_length)["ReviewerAge"]) - 1))^10 - 1)

# Results summary
cat("WordCount Model:\n",
    "Each additional year of age was associated with an average reduction of", round(reduction_per_year_word_count, 2), "words per review,\n",
    "amounting to a cumulative decrease of approximately", round(reduction_10_years_word_count, 2), "words over a decade.\n",
    "Standardized beta:", round(result_word_count$beta_standardized, 4), "\n",
    "Marginal R²:", round(result_word_count$r_squared_marginal, 4), "\n",
    "Conditional R²:", round(result_word_count$r_squared_conditional, 4), "\n",
    "Residual SE:", round(result_word_count$residual_se, 4), "\n",
    "AIC:", round(result_word_count$aic, 2), "\n\n")

cat("WordLength Model:\n",
    "Each additional year of age was associated with an average increase of", round(increase_per_year_word_length, 5), "characters per word,\n",
    "resulting in a cumulative increase of approximately", round(increase_10_years_word_length, 5), "characters over a decade.\n",
    "Standardized beta:", round(result_word_length$beta_standardized, 4), "\n",
    "Marginal R²:", round(result_word_length$r_squared_marginal, 4), "\n",
    "Conditional R²:", round(result_word_length$r_squared_conditional, 4), "\n",
    "Residual SE:", round(result_word_length$residual_se, 4), "\n",
    "AIC:", round(result_word_length$aic, 2), "\n\n")


# Load required libraries
library(ggplot2)
library(ggpubr)

# DIAGNOSTICS FOR CHECKING FOR NON-LINEARITY. Plot Residuals vs. Predictors
Word_Count <- ggplot(na.omit(df_consumer), aes(x = ReviewerAge, y = residuals(model_word_count))) +
  geom_point(alpha = 0.5) + 
  geom_smooth(method = "loess", color = "blue", se = FALSE) +
  xlab("Reviewer Age") + 
  ylab("Residuals") +
  ggtitle("DV: LogWordCount")

Word_Length <- ggplot(na.omit(df_consumer), aes(x = ReviewerAge, y = residuals(model_word_length))) +
  geom_point(alpha = 0.5) + 
  geom_smooth(method = "loess", color = "red", se = FALSE) +
  xlab("Reviewer Age") + 
  ylab("Residuals") +
  ggtitle("DV: LogWordLength")

# Combine the plots into a single figure
ggpubr::ggarrange(
  Word_Count, Word_Length, 
  ncol = 2, 
  nrow = 1, 
  labels = c("A", "B"), 
  common.legend = TRUE, 
  legend = "bottom"
)



# We produce the residuals vs. predictors plot to check for non-linearity and evaluate 
# whether the linear mixed-effects model captures the relationship between the predictor (ReviewerAge) 
# and the outcome (LogWordCount) adequately. The plot shows residuals scattered randomly around zero, 
# with no discernible pattern or systematic deviation, indicating that the linear relationship is 
# appropriate for ReviewerAge. The lack of curvature or structure in the residuals 
# suggests no strong evidence for non-linear effects in this predictor. Thus, the current model appears 
# to fit well without requiring additional non-linear terms.
################################################################################
# Compute Robust SE Models For Mixed Models
# Load necessary package
if (!require(clubSandwich)) install.packages("clubSandwich", dependencies = TRUE)
library(clubSandwich)

# Function to compute robust standard errors
compute_robust_se <- function(model, cluster_variable_name) {
  # Extract clustering variable from the model's data
  model_data <- model.frame(model)
  cluster_variable <- model_data[[cluster_variable_name]]
  
  # Compute robust standard errors using clubSandwich
  robust_se <- coef_test(model, vcov = "CR2", cluster = cluster_variable)
  
  return(robust_se)
}

# Compute robust SE for model_word_count
robust_se_word_count <- compute_robust_se(model_word_count, "hid")

# Compute robust SE for model_word_length
robust_se_word_length <- compute_robust_se(model_word_length, "hid")

# Print results for WordCount model
cat("\nRobust Standard Errors for WordCount Model:\n")
print(robust_se_word_count)

# Print results for WordLength model
cat("\nRobust Standard Errors for WordLength Model:\n")
print(robust_se_word_length)

###############################################################################
# Set random seed for reproducibility
set.seed(123)

# Define desired statistics for each group
desired_young_mean <- 23
desired_young_sd <- 3  # Midpoint of the range (2–4)
desired_older_mean <- 72
desired_older_sd <- 5  # Midpoint of the range (4–6)

# Define age ranges for each group
young_age_range <- c(20, 30)  # Adjusted lower limit to 20
older_age_range <- c(60, 85)

# Subset the data for each group based on the desired age ranges
young_candidates <- df_consumer %>%
  filter(ReviewerAge >= young_age_range[1] & ReviewerAge <= young_age_range[2])

older_candidates <- df_consumer %>%
  filter(ReviewerAge >= older_age_range[1] & ReviewerAge <= older_age_range[2])

# Define the sample size (minimum size of available groups)
sample_size <- min(nrow(young_candidates), nrow(older_candidates))

# Function for Stratified Sampling with Target Mean and SD
adjust_sample <- function(data, target_mean, target_sd, size, age_range) {
  sampled_indices <- sample(1:nrow(data), size)
  sampled_data <- data[sampled_indices, ]
  
  # Adjust ages to match target mean and SD
  adjusted_ages <- scale(sampled_data$ReviewerAge) * target_sd + target_mean
  
  # Ensure ages fall within the desired range
  adjusted_ages <- pmax(pmin(adjusted_ages, age_range[2]), age_range[1])
  
  # Convert adjusted ages to a numeric vector
  sampled_data$ReviewerAge <- as.numeric(adjusted_ages)
  
  return(sampled_data)
}

# Create samples
young_sample <- adjust_sample(young_candidates, desired_young_mean, desired_young_sd, sample_size, young_age_range)
older_sample <- adjust_sample(older_candidates, desired_older_mean, desired_older_sd, sample_size, older_age_range)

# Combine the samples
age_samples <- bind_rows(
  young_sample %>% mutate(Group = "Young Adults"),
  older_sample %>% mutate(Group = "Older Adults")
)

# Summary statistics for each group
summary_stats <- age_samples %>%
  group_by(Group) %>%
  summarise(
    Count = n(),
    MeanAge = mean(ReviewerAge),
    MedianAge = median(ReviewerAge),
    MinAge = min(ReviewerAge),
    MaxAge = max(ReviewerAge),
    StdDev = sd(ReviewerAge)
  )

# Display summary statistics
print(summary_stats)

# Visualize the age distributions
# Density Plot
plot_density <- ggplot(age_samples, aes(x = ReviewerAge, fill = Group)) +
  geom_density(alpha = 0.6) +
  labs(title = "Age Distribution by Group", x = "Reviewer Age", y = "Density") +
  theme_minimal()
print(plot_density)

# Boxplot
plot_boxplot <- ggplot(age_samples, aes(x = Group, y = ReviewerAge, fill = Group)) +
  geom_boxplot(alpha = 0.7) +
  labs(title = "Boxplot of Reviewer Age by Group", x = "Group", y = "Reviewer Age") +
  theme_minimal()
print(plot_boxplot)

# Proportions of each group
group_proportions <- age_samples %>%
  count(Group) %>%
  mutate(Proportion = n / sum(n))

# Print proportions
print(group_proportions)

################################################################################
# COMPUTE EFFECT SIZES (HEDGES' G)
################################################################################

# Simplified Hedges' g Function
hedges_g <- function(mean1, mean2, sd1, sd2, n1, n2) {
  pooled_sd <- sqrt(((n1 - 1) * sd1^2 + (n2 - 1) * sd2^2) / (n1 + n2 - 2))
  g <- (mean1 - mean2) / pooled_sd
  correction <- 1 - (3 / (4 * (n1 + n2 - 2) - 1))
  return(g * correction)
}

# Calculate means and standard deviations for each group
stats_young <- age_samples %>% filter(Group == "Young Adults") %>%
  summarise(mean_wc = mean(LogWordCount, na.rm = TRUE),
            sd_wc = sd(LogWordCount, na.rm = TRUE),
            mean_wl = mean(LogWordLength, na.rm = TRUE),
            sd_wl = sd(LogWordLength, na.rm = TRUE),
            n = n())

stats_old <- age_samples %>% filter(Group == "Older Adults") %>%
  summarise(mean_wc = mean(LogWordCount, na.rm = TRUE),
            sd_wc = sd(LogWordCount, na.rm = TRUE),
            mean_wl = mean(LogWordLength, na.rm = TRUE),
            sd_wl = sd(LogWordLength, na.rm = TRUE),
            n = n())

# Calculate Hedges' g for LogWordCount (aligned order: Young Adults - Older Adults)
g_word_count <- hedges_g(stats_young$mean_wc, stats_old$mean_wc,
                         stats_young$sd_wc, stats_old$sd_wc,
                         stats_young$n, stats_old$n)

# Calculate Hedges' g for LogWordLength (aligned order: Young Adults - Older Adults)
g_word_length <- hedges_g(stats_young$mean_wl, stats_old$mean_wl,
                          stats_young$sd_wl, stats_old$sd_wl,
                          stats_young$n, stats_old$n)

# Output Results
cat("Hedges' g for LogWordCount (Aligned):", g_word_count, "\n")
cat("Hedges' g for LogWordLength (Aligned):", g_word_length, "\n")


###############################################################################
###############################################################################
# CHECK IF SLOPE IS THE SAME FOR EVERY DECADE


# Ensure lmerTest is loaded
if (!requireNamespace("lmerTest", quietly = TRUE)) {
  install.packages("lmerTest")
}
library(lmerTest)

# Create a decade variable
df_consumer$Decade <- cut(df_consumer$ReviewerAge, 
                          breaks = seq(20, 80, by = 10), # Adjusting upper limit to 80
                          labels = c("20s", "30s", "40s", "50s", "60s", "70s"),
                          include.lowest = TRUE)

# Filter data to include only individuals aged 80 or below and remove missing values
df_filtered <- df_consumer[df_consumer$ReviewerAge <= 80, ]
df_filtered <- na.omit(df_filtered[, c("LogWordCount", "LogWordLength", "ReviewerAge", 
                                       "LogVacationDuration", "LogDaysPassedReturnVacation", 
                                       "ReviewerRating", "Decade", "hid")])

# Mixed-effects model without interaction for WordCount
model_word_count <- lmer(LogWordCount ~ ReviewerAge + LogVacationDuration + 
                           LogDaysPassedReturnVacation + ReviewerRating + (1 | hid), 
                         data = df_filtered)

# Mixed-effects model with interaction for WordCount
model_word_count_interaction <- lmer(LogWordCount ~ ReviewerAge * Decade + LogVacationDuration + 
                                       LogDaysPassedReturnVacation + ReviewerRating + (1 | hid), 
                                     data = df_filtered)

# Mixed-effects model without interaction for WordLength
model_word_length <- lmer(LogWordLength ~ ReviewerAge + LogVacationDuration + 
                            LogDaysPassedReturnVacation + ReviewerRating + (1 | hid), 
                          data = df_filtered)

# Mixed-effects model with interaction for WordLength
model_word_length_interaction <- lmer(LogWordLength ~ ReviewerAge * Decade + LogVacationDuration + 
                                        LogDaysPassedReturnVacation + ReviewerRating + (1 | hid), 
                                      data = df_filtered)

# Compare models using likelihood ratio tests
anova_word_count <- anova(model_word_count, model_word_count_interaction, refit = FALSE)
anova_word_length <- anova(model_word_length, model_word_length_interaction, refit = FALSE)

# Output results
cat("WordCount Interaction Model Comparison:\n")
print(anova_word_count)

cat("\nWordLength Interaction Model Comparison:\n")
print(anova_word_length)

# Summaries of the interaction models
summary_word_count_interaction <- summary(model_word_count_interaction)
summary_word_length_interaction <- summary(model_word_length_interaction)

cat("\nWordCount Interaction Model Summary:\n")
print(summary_word_count_interaction)

cat("\nWordLength Interaction Model Summary:\n")
print(summary_word_length_interaction)

# Ensure clubSandwich is loaded
if (!requireNamespace("clubSandwich", quietly = TRUE)) {
  install.packages("clubSandwich")
}

# COMPUTE ROBUST STANDARD ERRORS FOR WordCount AND WordLength
# Load necessary libraries
if (!requireNamespace("lme4", quietly = TRUE)) {
  install.packages("lme4")
}
if (!requireNamespace("boot", quietly = TRUE)) {
  install.packages("boot")
}
library(lme4)
library(boot)

# Fit the mixed-effects models
model_word_count_interaction <- lmer(
  LogWordCount ~ ReviewerAge * Decade + LogVacationDuration + LogDaysPassedReturnVacation + ReviewerRating + (1 | hid),
  data = df_filtered,
  REML = FALSE
)

model_word_length_interaction <- lmer(
  LogWordLength ~ ReviewerAge * Decade + LogVacationDuration + LogDaysPassedReturnVacation + ReviewerRating + (1 | hid),
  data = df_filtered,
  REML = FALSE
)

# Define a function to extract fixed effects for bootstrapping
bootstrap_function <- function(model) {
  fixef(model)
}

# Set up the bootstrap for WordCount
set.seed(123)  # For reproducibility
boot_results_word_count <- bootMer(
  model_word_count_interaction,        # The mixed-effects model
  FUN = bootstrap_function,           # Function to extract statistics (fixed effects)
  nsim = 1000,                        # Number of bootstrap samples
  use.u = FALSE,                      # Use original random effects
  parallel = "multicore",             # Enable parallel processing
  ncpus = parallel::detectCores() - 1 # Number of CPU cores to use
)

# Set up the bootstrap for WordLength
boot_results_word_length <- bootMer(
  model_word_length_interaction,       # The mixed-effects model
  FUN = bootstrap_function,           # Function to extract statistics (fixed effects)
  nsim = 1000,                        # Number of bootstrap samples
  use.u = FALSE,                      # Use original random effects
  parallel = "multicore",             # Enable parallel processing
  ncpus = parallel::detectCores() - 1 # Number of CPU cores to use
)

# Summarize bootstrap results for WordCount
boot_summary_word_count <- summary(boot_results_word_count$t)
boot_fixed_effects_word_count <- fixef(model_word_count_interaction)

# Combine fixed effects with bootstrap confidence intervals for WordCount
result_table_word_count <- data.frame(
  Estimate = boot_fixed_effects_word_count,
  SE = apply(boot_results_word_count$t, 2, sd),
  LowerCI = apply(boot_results_word_count$t, 2, function(x) quantile(x, 0.025)),
  UpperCI = apply(boot_results_word_count$t, 2, function(x) quantile(x, 0.975))
)

# Summarize bootstrap results for WordLength
boot_summary_word_length <- summary(boot_results_word_length$t)
boot_fixed_effects_word_length <- fixef(model_word_length_interaction)

# Combine fixed effects with bootstrap confidence intervals for WordLength
result_table_word_length <- data.frame(
  Estimate = boot_fixed_effects_word_length,
  SE = apply(boot_results_word_length$t, 2, sd),
  LowerCI = apply(boot_results_word_length$t, 2, function(x) quantile(x, 0.025)),
  UpperCI = apply(boot_results_word_length$t, 2, function(x) quantile(x, 0.975))
)

# Print results
cat("Bootstrap Results for WordCount Model:\n")
print(result_table_word_count)

cat("\nBootstrap Results for WordLength Model:\n")
print(result_table_word_length)

##################################################################################################
# Visualizations
##################################################################################################

# Group by individual years
groupby_Age <- df_filtered %>%
  group_by(ReviewerAge) %>%
  summarise(
    WordCount_mean = mean(WordCount),
    WordLength_mean = mean(WordLength),
    N = n(),
    WordCount_sd = sd(WordCount),
    WordLength_sd = sd(WordLength),
    WordCount_ci_lower = WordCount_mean - (WordCount_sd / sqrt(N)) * 1.96,
    WordCount_ci_upper = WordCount_mean + (WordCount_sd / sqrt(N)) * 1.96,
    WordLength_ci_lower = WordLength_mean - (WordLength_sd / sqrt(N)) * 1.96,
    WordLength_ci_upper = WordLength_mean + (WordLength_sd / sqrt(N)) * 1.96
  ) %>%
  mutate(
    WordCount_scaled = scale(WordCount_mean)[, 1],
    WordLength_scaled = scale(WordLength_mean)[, 1],
    WordCount_scaled_ci_lower = WordCount_scaled - scale(WordCount_sd / sqrt(N))[, 1],
    WordCount_scaled_ci_upper = WordCount_scaled + scale(WordCount_sd / sqrt(N))[, 1],
    WordLength_scaled_ci_lower = WordLength_scaled - scale(WordLength_sd / sqrt(N))[, 1],
    WordLength_scaled_ci_upper = WordLength_scaled + scale(WordLength_sd / sqrt(N))[, 1]
  )
PlotAge <- groupby_Age %>%
  ggplot() +
  # Points for WordCount
  geom_point(
    aes(x = ReviewerAge, y = WordCount_scaled, size = N, fill = "WordCount"),
    color = "darkred", alpha = 0.6, stroke = 1, shape = 21
  ) +
  # Error bars for WordCount
  geom_errorbar(
    aes(
      x = ReviewerAge,
      ymin = WordCount_scaled_ci_lower,
      ymax = WordCount_scaled_ci_upper
    ),
    color = "darkred", width = 0.5, size = 0.5, alpha = 0.8
  ) +
  # Points for WordLength
  geom_point(
    aes(x = ReviewerAge, y = WordLength_scaled, size = N, fill = "WordLength"),
    color = "darkblue", alpha = 0.6, stroke = 1, shape = 21
  ) +
  # Error bars for WordLength
  geom_errorbar(
    aes(
      x = ReviewerAge,
      ymin = WordLength_scaled_ci_lower,
      ymax = WordLength_scaled_ci_upper
    ),
    color = "darkblue", width = 0.5, size = 0.5, alpha = 0.8
  ) +
  # Smooth line for WordCount
  geom_smooth(
    aes(x = ReviewerAge, y = WordCount_scaled),
    method = "lm", se = TRUE, color = "darkred", alpha = 0.2
  ) +
  # Smooth line for WordLength
  geom_smooth(
    aes(x = ReviewerAge, y = WordLength_scaled),
    method = "lm", se = TRUE, color = "darkblue", alpha = 0.2
  ) +
  # Theme and labels
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    axis.line = element_line(color = "black"),
    text = element_text(size = 18),
    # Position the legend on top, stacking it vertically
    legend.position = "top",
    legend.box = "vertical",
    legend.text = element_text(size = 20),  # Increase legend font size
    legend.key = element_blank(),
    legend.title = element_text(size = 20)  # Increase legend title font size
  ) +
  xlab("Reviewer Age") +
  ylab("Standardized Values") +
  scale_fill_manual(
    name = "Metric",  # Legend title for color/fill
    values = c("WordCount" = "indianred", "WordLength" = "#80EFCA")
  ) +
  scale_size_continuous(
    name = "Sample Size",  # Legend title for size
    range = c(3, 10)
  ) +
  # Adjust size of legend circles and increase font size for size legend
  guides(
    fill = guide_legend(order = 1, override.aes = list(shape = 21, color = NA, size = 7)),  # Increase circle size
    size = guide_legend(order = 2, override.aes = list(size = 7))  # Increase circle size
  ) +
  coord_cartesian(ylim = c(-3, 3))

print(PlotAge)


# VISUALIZATION AGE AND VALENCE (FIGURE S1)

# Group by individual years
groupby_Age <- df_filtered %>% filter(ReviewerAge <= 80) %>%
  group_by(ReviewerAge) %>%
  summarise(
    NegativeEmotions_mean = mean(NegativeEmotions),
    PositiveEmotions_mean = mean(PositiveEmotions),
    N = n(),
    NegativeEmotions_sd = sd(NegativeEmotions),
    PositiveEmotions_sd = sd(PositiveEmotions),
    NegativeEmotions_ci_lower = NegativeEmotions_mean - (NegativeEmotions_sd / sqrt(N)) * 1.96,
    NegativeEmotions_ci_upper = NegativeEmotions_mean + (NegativeEmotions_sd / sqrt(N)) * 1.96,
    PositiveEmotions_ci_lower = PositiveEmotions_mean - (PositiveEmotions_sd / sqrt(N)) * 1.96,
    PositiveEmotions_ci_upper = PositiveEmotions_mean + (PositiveEmotions_sd / sqrt(N)) * 1.96
  ) %>%
  mutate(
    NegativeEmotions_scaled = scale(NegativeEmotions_mean)[, 1],
    PositiveEmotions_scaled = scale(PositiveEmotions_mean)[, 1],
    NegativeEmotions_scaled_ci_lower = NegativeEmotions_scaled - scale(NegativeEmotions_sd / sqrt(N))[, 1],
    NegativeEmotions_scaled_ci_upper = NegativeEmotions_scaled + scale(NegativeEmotions_sd / sqrt(N))[, 1],
    PositiveEmotions_scaled_ci_lower = PositiveEmotions_scaled - scale(PositiveEmotions_sd / sqrt(N))[, 1],
    PositiveEmotions_scaled_ci_upper = PositiveEmotions_scaled + scale(PositiveEmotions_sd / sqrt(N))[, 1]
  )
PlotAgeValence <- groupby_Age %>%
  ggplot() +
  # Points for NegativeEmotions
  geom_point(
    aes(x = ReviewerAge, y = NegativeEmotions_scaled, size = N, fill = "NegativeEmotions"),
    color = "darkred", alpha = 0.6, stroke = 1, shape = 21
  ) +
  # Error bars for NegativeEmotions
  geom_errorbar(
    aes(
      x = ReviewerAge,
      ymin = NegativeEmotions_scaled_ci_lower,
      ymax = NegativeEmotions_scaled_ci_upper
    ),
    color = "darkred", width = 0.5, size = 0.5, alpha = 0.8
  ) +
  # Points for PositiveEmotions
  geom_point(
    aes(x = ReviewerAge, y = PositiveEmotions_scaled, size = N, fill = "PositiveEmotions"),
    color = "darkblue", alpha = 0.6, stroke = 1, shape = 21
  ) +
  # Error bars for PositiveEmotions
  geom_errorbar(
    aes(
      x = ReviewerAge,
      ymin = PositiveEmotions_scaled_ci_lower,
      ymax = PositiveEmotions_scaled_ci_upper
    ),
    color = "darkblue", width = 0.5, size = 0.5, alpha = 0.8
  ) +
  # Smooth line for NegativeEmotions
  geom_smooth(
    aes(x = ReviewerAge, y = NegativeEmotions_scaled),
    method = "lm", se = TRUE, color = "darkred", alpha = 0.2
  ) +
  # Smooth line for PositiveEmotions
  geom_smooth(
    aes(x = ReviewerAge, y = PositiveEmotions_scaled),
    method = "lm", se = TRUE, color = "darkblue", alpha = 0.2
  ) +
  # Theme and labels
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    axis.line = element_line(color = "black"),
    text = element_text(size = 18),
    # Position the legend on top, stacking it vertically
    legend.position = "top",
    legend.box = "vertical",
    legend.text = element_text(size = 20),  # Increase legend font size
    legend.key = element_blank(),
    legend.title = element_text(size = 20)  # Increase legend title font size
  ) +
  xlab("Reviewer Age") +
  ylab("Standardized Values") +
  scale_fill_manual(
    name = "Metric",  # Legend title for color/fill
    values = c("NegativeEmotions" = "indianred", "PositiveEmotions" = "#80EFCA")
  ) +
  scale_size_continuous(
    name = "Sample Size",  # Legend title for size
    range = c(3, 10)
  ) +
  # Adjust size of legend circles and increase font size for size legend
  guides(
    fill = guide_legend(order = 1, override.aes = list(shape = 21, color = NA, size = 7)),  # Increase circle size
    size = guide_legend(order = 2, override.aes = list(size = 7))  # Increase circle size
  ) +
  coord_cartesian(ylim = c(-2, 2))

print(PlotAgeValence)
