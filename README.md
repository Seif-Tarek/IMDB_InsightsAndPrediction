# IMDB_InsightsAndPrediction
 

In 2019, movies made an estimated $42.5 billion revenue. Lately, the movies industry is 
growing so fast and more people are getting involved in it and investing huge amounts of money. 
New and current investors surely want to know if they are investing in the right movie or not, also 
they want an estimate of the revenue of the movie.

In the project we followed the regular data science pipeline for model building starting by 
visualizing and analysing the data, extracting insights from it and engineer features that we see 
helpful for predicting the movie revenue.

# Visualization and insights
• Visualizing the data sets and its structure.
• Gettin data insights that are useful for the problem solution
# Pre-processing and Feature engineering
• Fix the data formatting and structure for feature extraction
• Manipulate the data to add and transform features 
# Model building 
• Using the extracted features, iteratively build models that attemt to solve our problem
• go back to the feature engineering step with feedback

The dataset that we chose is ‘TMDB movies dataset’ which is a movies dataset collected by 
the famous TMDB (The movies database) website.

We have finally settled on a regression model called ‘Light GBM’ (Light Gradient Boosting  Machine - LGBM), after many unsuccessful trials and experiments which will be mentioned in the  next section, which is basically an algorithm from the boosting techniques like the famous and  simple AdaBoost, but the model that it uses to combine weak predictors and estimators in-order to get a strong estimator is, the decision tree regression model. The idea is that it uses a predefined number of estimators from the decision tree models, and combines the weak learners to get a strong one.

The root mean-square error (RMSE) after training the former model, LGBM, Training RMSE = 1.6365, Validation RMSE = 1.965156.

