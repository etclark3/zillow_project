# Zillow SFH Price Prediction Project
--------------
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a>
    <li><a href="#summary">Summary</a></li>
    <li><a href="#data-dictionary">Data Dictionary</a></li>
    <li><a href="#preliminary-questions">Questions</a></li>
    <li><a href="#planning">Planning</a></li>
    <li><a href="#Conclusion-and-Recommendations">Conclusion and Recommendations</a></li>
    <li><a href="#additional-improvements">Additional Improvements</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#for-reproduction">For Reproduction</a></li>
  </ol>
</details>
    
# About/Goals:
- Using Linear Regession Models on Zillow data, attempt to accurately predict Single Family Home (SFH) prices for homes sold in 2017

# Summary:
- Seven features (bathrooms, bedrooms, bldg_quality, f_sqft, fips, lot_size, zip) derived from the Zillow data with originally 62 feature columns were the most accurate to predict the Tax Value (tax_value) of a SFH.
- The Ordinary Least Squared Regression model predicted the most accurately out of three models and was approximately a 40,000 improvement in the standard deviation of variance from the mean and median predictions.
- The model's R-squared scored was 6% higher than any other model used and both RMSE and R-squared were higher on the test data than the train data.

# Preliminary Questions:
    - What are the most important features that determine a home's value?
    - How many homes are outside the ordinary from a "regular home"?
    - How much does location play a part in value?
# Planning:
![Zillow_pipeline](https://user-images.githubusercontent.com/98612085/189562019-2c09bdbf-5358-4be3-98e6-91cf3b124af3.png)

# Data Dictionary:

Zillow Data Features:

<img width="1187" alt="image" src="https://user-images.githubusercontent.com/98612085/189556700-b97d7450-bafa-47f8-81a5-10377050600a.png">

<img width="1062" alt="image" src="https://user-images.githubusercontent.com/98612085/189556238-f433cb25-1158-4a29-91bd-4f2b9dc58c55.png">

# Conclusion and Recommendations: 
Square footage of the property
Number of Bedrooms and Bathrooms
Location

Using Recursive Reature Elimination (RFE) 

Takeaways


Model - The Ordinary Least Squared regression model had the best performance.

- Train: 
  - 205,837 RMSE
  - 0.246 R-squared value
- Validate: 
  - 205,837 RMSE
  - 0.246 R-squared value
- Test: 
  - 205,837 RMSE
  - 0.28 R-squared value
  - 
# Next Steps/Reccomendations: 

Following this initial project, I would like to create a more refined model. Whether it is through running different groups of features through my current model, tuning the hyperparameters or both.
As well as creating more python files that store functions so as to make the final report less filled by code and easier to read.
Create a feature that clusters locations in the dataset, most home prices fluctuate by location within cities. With certain city locations being higher in crime, less wealthy, under developed, etc.
Include previous assesments of the porperties to see if previous pricing of properties affect sell price.

# Additional Improvements:
- Remove additional outliers and focus data on "normal homes" to increase accuracy of model predictions for homes for the 2nd and 3rd quartile of data
- Use census tract and block data as they define subdivision bounds. Homes values are typically similar in neighborhoods

# Contact:
Everett Clark - everett.clark.t@gmail.com

# For Reproduction:
First you will need database server credentials, then:

- Download wrangle.py and project_final_notebook
- Add your own credentials to the directory (username, host, password)
- Run the project_final_notebook

[[Back to top](#top)]
