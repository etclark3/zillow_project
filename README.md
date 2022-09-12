# Zillow SFH Price Prediction Project

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about">About</a>
    <li><a href="#summary">Summary</a></li>
    <li><a href="#goals">Goals</a></li>
    <li><a href="#data-dictionary">Data Dictionary</a></li>
    <li><a href="#additional-improvements">Additional Improvements</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>
    
# About:
- Using Linear Regession Models on Zillow data, attempt to accurately predict Single Family Home (SFH) prices for homes sold in 2017

# Summary:
- Seven features (bathrooms, bedrooms, bldg_quality, f_sqft, fips, lot_size, zip) derived from the Zillow data with originally 62 feature columns were the most accurate to predict the Tax Value (tax_value) of a SFH.
- The Ordinary Least Squared Regression model predicted the most accurately out of three models and was approximately a 40,000 improvement in the standard deviation of variance from the mean and median predictions.
- The model's R-squared scored was 6% higher than any other model used and both RMSE and R-squared were higher on the test data than the train data.

# Goals:
-----------
# Data Dictionary:

Zillow Data Features:

<img width="1187" alt="image" src="https://user-images.githubusercontent.com/98612085/189556700-b97d7450-bafa-47f8-81a5-10377050600a.png">

<img width="1062" alt="image" src="https://user-images.githubusercontent.com/98612085/189556238-f433cb25-1158-4a29-91bd-4f2b9dc58c55.png">

# Recommendations: 
# Additional Improvements:
- Remove additional outliers and focus data on "normal homes" to increase accuracy of model predictions for homes for the 2nd and 3rd quartile of data
- Use census tract and block data as they define subdivision bounds. Homes values are typically similar in neighborhoods

# Contact:
Everett Clark - everett.clark.t@gmail.com

[[Back to top](#top)]
