# Zillow SFH Price Prediction Project

# Overview:
- Using Linear Regession Models, accurately predict SFH prices
- 

# Goals:
-----------
# Data Dictionary:

Zillow Data Features:

<img width="586" alt="image" src="https://user-images.githubusercontent.com/98612085/189434891-0d333546-6f4f-48a9-9287-673b05b27639.png">
Feature	Null Count	Drop/Not	Reason	Description
transactiondate:	0	Drop	Filtered down to same year (2017)	Latest date of transaction for the property
Month	0	N	Derived from transactiondate	Added to take the place of transaction date
propertylandusedesc:	0	Drop	All SFH(Code 261)	Code for what type of home is on the land
airconditioningtypeid:	38803	Drop	Too many nulls	Type of cooling system present in the home (if any)
architecturalstyletypeid:	52371	Drop	Too many nulls	Architectural style of the home (i.e. ranch, colonial, split-level, etc…)
basementsqft:	52394	Drop	Too many nulls	Finished living area below or partially below ground level
bathroomcnt:	0	N	Vital to home value	Number of bathrooms in home including fractional bathrooms
bedroomcnt:     	0	N	Vital to home value	Number of bedrooms in home
buildingqualitytypeid:	18701	N	Material quality used in homes matters	Overall assessment of condition of the building from best (lowest) to worst (highest)
buildingclasstypeid:	52441	Drop	Too many nulls	The building framing type (steel frame, wood frame, concrete/brick)
calculatedbathnbr:	137	Drop	Almost identical to bathroomcnt	Number of bathrooms in home including fractional bathroom
decktypeid:	52052	Drop	Too many nulls	Type of deck (if any) present on parcel
threequarterbathnbr:	45717	Drop	Too many nulls	Number of 3/4 bathrooms in house (shower + sink + toilet)
finishedfloor1squarefeet: 	48060	Drop	Too many nulls	Size of the finished living area on the first (entry) floor of the home
calculatedfinishedsquarefeet:	82	N	Vital to home value	Calculated total finished living area of the home
finishedsquarefeet6:	52276	Drop	Too many nulls	Base unfinished and finished area
finishedsquarefeet12:	247	Drop	too many similarites to calculatedfinishedsquarefeet	Finished living area
finishedsquarefeet13:	52441	Drop	Too many nulls	Perimeter living area
finishedsquarefeet15:	52441	Drop	Too many nulls	Total area
finishedsquarefeet50:	48060	Drop	Too many nulls	Size of the finished living area on the first (entry) floor of the home
fips:	0	N	County Identifier	Federal Information Processing Standard code 
fireplacecnt:	45198	Drop	Too many nulls	Number of fireplaces in a home (if any)
fireplaceflag:	52360	Drop	Too many nulls	Is a fireplace present in this home
fullbathcnt:	137	Drop	Not needed	Number of full bathrooms (sink, shower + bathtub, and toilet) present in home
garagecarcnt:	34426	Drop	Too many nulls, not going to imput 50%	Total number of garages on the lot including an attached garage
garagetotalsqft:	34426	Drop	Not needed	Total number of square feet of all garages on lot including an attached garage
hashottuborspa:	50926	Drop	Too many nulls	Does the home have a hot tub or spa
heatingorsystemtypeid:	18506	N	Maybe system in home matters	Type of home heating system
latitude:	0	N	Location is key	Latitude of the middle of the parcel multiplied by 10e6
longitude:	0	N	Location is key	Longitude of the middle of the parcel multiplied by 10e6
lotsizesquarefeet:	369	N	Believe lot size will play a part	Area of the lot in square feet
numberofstories:	37880	Drop	Too many nulls and doesn't usually affect price 	Number of stories or levels the home has
parcelid:	0	Drop	Identifier (moved to separate df)	Unique identifier for parcels (lots)
poolcnt:	41345	Drop	Too many nulls	Number of pools on the lot (if any)
poolsizesum:	51574	Drop	Too many nulls	Total square footage of all pools on property
pooltypeid10:	51997	Drop	Too many nulls	Spa or Hot Tub
pooltypeid2:	51370	Drop	Too many nulls	Pool with Spa/Hot Tub
pooltypeid7: 	42432	Drop	Too many nulls	Pool without hot tub
propertycountylandusecode:	0	Drop	All zoned for SFH	County land use code i.e. it's zoning at the county level
propertylandusetypeid:	0	Drop	All the same type	Type of land use the property is zoned for
propertyzoningdesc:	18593	Drop	All zone for SFH, not commercial	Description of the allowed land uses (zoning) for that property
rawcensustractandblock:	0	Drop	Focused on other features	Census tract and block ID combined - also contains blockgroup assignment by extension
censustractandblock:	123	Drop	Focused on other features	Census tract and block ID combined - also contains blockgroup assignment by extension
regionidcounty:	0	Drop	Already have county data(FIPS)	County in which the property is located
regionidcity:	1037	Drop	regionidzip has less nulls and more diversified values	City in which the property is located (if any)
regionidzip:	26	N	Location is key	Zip code in which the property is located
regionidneighborhood:	33408	 Drop	Many nulls, I have other location features	Neighborhood in which the property is located
roomcnt:	0	Drop	Too many incorrect values, 0 rooms for over half the dataset	Total number of rooms in the principal residence
storytypeid:	52394	Drop	Too many nulls	Type of floors in a multi-story house (i.e. basement and main level, split-level, attic, etc.)
typeconstructiontypeid:	52365		Too many nulls	What type of construction material was used to construct the home
unitcnt:	18594	Drop	99% of non-nulls had only 1 unit	Number of units the structure is built into (i.e. 2 = duplex, 3 = triplex, etc...)
yardbuildingsqft17:	50504	Drop	Too many nulls	Patio in yard
yardbuildingsqft26:	52378	Drop	Too many nulls	Storage shed/building in yard
yearbuilt:	116	N	Year built may have sway in value	The Year the principal residence was built
taxvaluedollarcnt:	1	N	Target	The total tax assessed value of the parcel
structuretaxvaluedollarcnt:	84	Drop	Potentially derived from target	The assessed value of the built structure on the parcel
landtaxvaluedollarcnt:	1	Drop	Potentially derived from target	The assessed value of the land area of the parcel
taxamount:	4	Drop	Potentially derived from target	The total property tax assessed for that assessment year
assessmentyear:	0	Drop	All 2016	The year of the property tax assessment
taxdelinquencyflag:	50362	Drop	Too many nulls	Property taxes for this parcel are past due as of 2015
taxdelinquencyyear:	50362	Drop	Too many nulls	Year for which the unpaid propert taxes were due
