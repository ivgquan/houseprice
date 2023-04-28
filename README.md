# **I. INTRODUCTION**

**A. Datasets**

**The first 3 data sets are FED economic data:** this is from Federal Reserve Bank Of St. Louis (FED) to show the economic conditions in the US.

* CPIAUCSL: **The US CPI** to measure the inflation, reported Monthly

*  RRVRUSQ156N: **The rental vacancy rate** to measure the percentage of rental properties that are vacant in the US, reported Quarterly

* MORTGAGE30US: **The mortgage interest rates** in the US , reported Weekly

**The last 2 data sets are Zillow price data**: this is from Zillow - a website provide data about real estate and homes for sale in US.

* Metro_median_sale_price_uc_sfrcondo_week: **The median sale price of houses** in US, reported Weekly

* Metro_zhvi_uc_sfrcondo_tier_0:  **The median value of all houses in US computed by Zillow**, reported Monthly

**B. Goal**
- Prediction will be the trend of houses price whether it goes up or down in the next reported period

# **II. Data Processing**
**A. Handle FED economic data**
Parse any dates in the csv file into pandas's date time and use first column as index
```php
fed_files = ["MORTGAGE30US.csv", "RRVRUSQ156N.csv", "CPIAUCSL.csv"]
dfs = [pd.read_csv(f, parse_dates=True, index_col=0) for f in fed_files]
```
**It can be seen that** 
* The mortgage interest rates is reported **Weekly**
* The rental vacancy rate is reported **Quarterly**
* The US CPI is reported **Monthly**
![image](https://user-images.githubusercontent.com/131565330/235098928-d0cd2225-f905-4398-ab68-5fbd9560113c.png)
I will **Combine 3 dataframes to 1 big dataframe** and we can see NaN because those are reported in different time **(Weekly, Quarterly and Monthly)**
```php
fed_data = pd.concat(dfs, axis=1)
fed_data.tail(7)
```
![image](https://user-images.githubusercontent.com/131565330/235099359-c263c408-1f8d-446b-8716-2234934cc0ce.png)

**To fix this issue,** I will assume that these rates are going to stay constant for the period in which they are released by using **forward filling**
For example, **the US CPI (3rd Column)** is released **Monthly** so I will assume that **295.271** will stay constant for the whole month **(July,2022)**
```php
fed_data= fed_data.ffill()
fed_data= fed_data.dropna()#drop missing value that dont have all 3 economic indicators
fed_data.tail(7)
```
![image](https://user-images.githubusercontent.com/131565330/235099274-d1083ad6-684f-42cf-9576-34d59d776f54.png)

**B. Handle Zillow price data**
```php
zillow_files = ["Metro_median_sale_price_uc_sfrcondo_week.csv", "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv"]
dfs = [pd.read_csv(f) for f in zillow_files]
```
**In these 2 dataset**, we can see that
* Each rows is a region in the US
* Columns shows information about the region
![image](https://user-images.githubusercontent.com/131565330/235099521-a3a8aacf-3095-45a1-a81c-958565d0a49b.png)

**In both 2 dataframes:**

* I will only **take the first row** because I only want to analyze data from the US 

* I also **remove these first 5 columns** to have only **The median sale price of house**, reported Weekly and **The house value computed by Zillow**, reported Monthly

```php
dfs = [pd.DataFrame(df.iloc[0,5:]) for df in dfs]
for df in dfs:
    df.index = pd.to_datetime(df.index)
    df["month"] = df.index.to_period("M")
 ```
 **Because data in these 2 datatframe are report at different period** (Weekly and Monthly) so that I will add a column to both to merge them together using the common column.
![image](https://user-images.githubusercontent.com/131565330/235099784-bbef3b5f-5467-4791-b1d7-f33bb783a278.png)

```php
price_data = dfs[0].merge(dfs[1], on="month") #merge using the common column
price_data.index = dfs[0].index #set index of this dataframe is the same as the index of the 1st dataframe
del price_data["month"] #Drop column month 
price_data.columns = ["price", "value"] #Change columns name
price_data
```
![image](https://user-images.githubusercontent.com/131565330/235099888-22cb88d5-922a-40be-a417-00c3afa6957e.png)

**C.** Merge **Zillow** price data with **FED** economic data
**Because FED released its data 2 days sooner than Zillow,** so I am going to add a couple of days(shift forward) to align Fed data with Zillow data
```php
from datetime import timedelta
fed_data.index = fed_data.index + timedelta(days=2)
price_data = fed_data.merge(price_data, left_index=True, right_index=True) 
#combine 2 df based on the matching of index which is the date in both df -> only took dates that matched in both df, anything that didnt match will be removed
price_data.columns = ["Interest rate", "Vacancy rate", "CPI", "Median Sale Price", "House Value"] #rename columns
price_data
```
![image](https://user-images.githubusercontent.com/131565330/235100063-bff69dce-8b51-417e-97db-2dec0e8320bf.png)

**D. Data Visualization Overview**
```php
price_data.plot.line(y="Median Sale Price", use_index=True, title='Median Sale Price with Inflation')
price_data.plot.line(y="adj_price", use_index=True, title='Median Sale Price without Inflation')
price_data["adj_price"] = price_data["Median Sale Price"] / price_data["CPI"] * 100 #create an adjust price which is not affected by inflation
price_data["adj_value"] = price_data["House Value"] / price_data["CPI"] * 100 #create an adjust value which is not affected by inflation
```
![image](https://user-images.githubusercontent.com/131565330/235101156-2269a48f-65e0-4e9d-b88d-807dd7d2d74d.png)

# **III. Set up Target**
I will try to predict what will happen to house prices next Quarter **(go up or go down 3months from now)** by using pandas shift method
```php
price_data["next_quarter"] = price_data["adj_price"].shift(-13) 
#shift method will grabs the adjusted price into the future and pulls it back to the current row
price_data
```
![image](https://user-images.githubusercontent.com/131565330/235101441-b116fb78-6def-4de8-b9da-da7568397640.png)

* Those rows that have NaN will be used to make predictions after building the model

* In order to train the algorithm, I have to actually know what happened so I could use these to make future predictions 

* Therefore, those rows can not be used for training data and I will drop them

```php
 price_data_need_to_predict = price_data.copy()
 save= price_data_need_to_predict.tail(13) #save data to predict later
 price_data.dropna(inplace=True)#drop na value
 price_data
 ```
 ![image](https://user-images.githubusercontent.com/131565330/235101882-a5be722a-712a-47cb-9d43-41e4993d3f32.png)

**I will add column 'Change' as a Target** to show whether the price go up or down in the next quarter
```php
price_data["change"] = (price_data["next_quarter"] > price_data["adj_price"]).astype(int)  #True =1, False=0 which means if the price goes up, change=1 and if the price goes down, change=0
#the column 'Change' will show the price 3 months from now is higher or lower than the current price in each row
price_data
```
![image](https://user-images.githubusercontent.com/131565330/235102066-0360d4c9-afe7-4fce-96f5-453c2ed7ca54.png)

Now I would want to know how many weeks did the price **go up** and how many weeks did the price **go down**
```php
price_data["change"].value_counts()
```
![image](https://user-images.githubusercontent.com/131565330/235102248-3e76f70b-e974-46cc-bd98-67a5511be5d7.png)

I will use variables (predictors) **to make prediction** and the target will be the 'Change' column  
```php
predictors = ["Interest rate", "Vacancy rate", "adj_price", "adj_value"] #use 4 columns to predict 1 column #features selection
target = "change"
```
# **IV. Build Model**
```php
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
#this will tell the accuracy percentage , when the model predicts the price would go up, how often did it go up and when the model predicts the price would go down how often did it go down
import numpy as np
```
**First I will create a prediction function**. This function will take in these inputs below and **return predictions**
* a training set 
* a test set which is what I want to make predictions on
* a set of predictor which are the columns I am going to use to make predictions
* a target

**Then I create a backtest function**. This function will let me generate predictions for most of my data set but do it in a way that
respects the order of the data set so that I can avoid using future data to predict the past
```php
START = 260 #start with 5 years of data, it will take all the data from 2008 to 2013 to predict 2014, then it will take all the data from 2008 to 2014 to predict 2015 and so on until I have  predictions for every year from 2014 through 2022 
STEP = 52 #52 weeks in 1 year
def predict(train, test, predictors, target):
    rf = RandomForestClassifier(min_samples_split=10, random_state=1) #min protects against overfitting by preventing the nodes in the decision trees in the random forest from splitting too deeply, a random state to ensure that every time I run my model it's going to generate the same sequence of random numbers, thereby giving the same result
    rf.fit(train[predictors], train[target]) #fit model using training data
    preds = rf.predict(test[predictors]) #generate predictions using test set
    return preds
def backtest(data, predictors, target):
    all_preds = [] #Create a list called all predictions
    for i in range(START, data.shape[0], STEP):
        train = price_data.iloc[:i] #everything up until i
        test = price_data.iloc[i:(i+STEP)] #the year following i
        all_preds.append(predict(train, test, predictors, target)) #append all of prediction sets to the list
    preds = np.concatenate(all_preds) #all predictions is going to be a list of numpy arrays (all_preds) -> concatenate those arrays together into a single array
    return preds, accuracy_score(data.iloc[START:][target], preds) # (data.iloc[START:][target]) are Actual values for test data vs (preds) Prediction values
```
![image](https://user-images.githubusercontent.com/131565330/235102599-e56b214c-bb80-4e52-a35c-48473aeddbb4.png)

Now, it shows that I have **59%** accuracy in my predictions

# **V. Improve Model**




