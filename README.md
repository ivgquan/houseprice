# **I. Introduction**

**A. Datasets overview**

![image](https://user-images.githubusercontent.com/131565330/235114630-14721b82-40de-4ef9-82c0-f21783958df6.png)


**The first 3 data sets are FED economic data:** this is from Federal Reserve Bank Of St. Louis (FED) to show the economic conditions in the US.

* CPIAUCSL:  **The US CPI** to measure the inflation, reported Monthly

*  RRVRUSQ156N: **The rental vacancy rate** to measure the percentage of rental properties that are vacant in the US, reported Quarterly

* MORTGAGE30US: **The mortgage interest rates** in the US , reported Weekly


**The last 2 data sets are Zillow price data**: this is from Zillow - a website provide data about real estate and homes for sale in US.

* Metro_median_sale_price_uc_sfrcondo_week: **The median sale price of houses** in US, reported Weekly

* Metro_zhvi_uc_sfrcondo_tier_0:  **The median value of all houses in US computed by Zillow**, reported Monthly

**B. My goal**
- Prediction will be the trend of houses price whether it goes up or down in the next reported period

# **II. Data processing**
**A. Handle FED economic data**

Parse any dates in the csv file into pandas's date time and use first column as index.
```php
fed_files = ["MORTGAGE30US.csv", "RRVRUSQ156N.csv", "CPIAUCSL.csv"]
dfs = [pd.read_csv(f, parse_dates=True, index_col=0) for f in fed_files]
```

**It can be seen that** 
* The mortgage interest rates is reported **Weekly**
* The rental vacancy rate is reported **Quarterly**
* The US CPI is reported **Monthly**

![image](https://user-images.githubusercontent.com/131565330/235098928-d0cd2225-f905-4398-ab68-5fbd9560113c.png)

I will **combine 3 dataframes to 1 big dataframe** and we can see NaN because those are reported in different time **(Weekly, Quarterly and Monthly)**.
```php
fed_data = pd.concat(dfs, axis=1)
fed_data.tail(7)
```

![image](https://user-images.githubusercontent.com/131565330/235099359-c263c408-1f8d-446b-8716-2234934cc0ce.png)

* **To fix this issue,** I will assume that these rates are going to stay constant for the period in which they are released by using **forward filling**.
* For example, **the US CPI (3rd Column)** is released **Monthly** so I will assume that **295.271** will stay constant for the whole month **(July,2022)**.

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

**In these 2 dataset**, we can see that:
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
Now I have **price_data**.

![image](https://user-images.githubusercontent.com/131565330/235099888-22cb88d5-922a-40be-a417-00c3afa6957e.png)

**C. Merge Zillow price data with FED economic data**

Because FED released its data 2 days sooner than Zillow, so I am going to add a couple of days **(shift forward)** to align Fed data with Zillow data.

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
I will remove Inflation to know clearly about the increase in house prices **(the Underlying house prices)**.
```php
price_data.plot.line(y="Median Sale Price", use_index=True, title='Median Sale Price with Inflation')
price_data.plot.line(y="adj_price", use_index=True, title='Median Sale Price without Inflation')
price_data["adj_price"] = price_data["Median Sale Price"] / price_data["CPI"] * 100 #create an adjust price which is not affected by inflation
price_data["adj_value"] = price_data["House Value"] / price_data["CPI"] * 100 #create an adjust value which is not affected by inflation
```

![image](https://user-images.githubusercontent.com/131565330/235101156-2269a48f-65e0-4e9d-b88d-807dd7d2d74d.png)

# **III. Set up target**

I will try to predict what will happen to house prices next Quarter **(go up or go down 3months from now)** by using pandas shift method.

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

**I will add column 'Change' as a Target** to show whether the price go up or down in the next quarter.
*   If the price of current row **goes up** in next quarter, **Change = 1**

*   If the price of current row **goes down** in next quarter, **Change = 0**

```php
price_data["change"] = (price_data["next_quarter"] > price_data["adj_price"]).astype(int)  #True =1, False=0 which means if the price goes up, change=1 and if the price goes down, change=0
#the column 'Change' will show the price 3 months from now is higher or lower than the current price in each row
price_data
```

![image](https://user-images.githubusercontent.com/131565330/235102066-0360d4c9-afe7-4fce-96f5-453c2ed7ca54.png)

Now I would want to know how many weeks did the price **go up** and how many weeks did the price **go down**.

```php
price_data["change"].value_counts()
```

![image](https://user-images.githubusercontent.com/131565330/235102248-3e76f70b-e974-46cc-bd98-67a5511be5d7.png)

I will use variables (predictors) **to make prediction** and the target will be the 'Change' column.

```php
predictors = ["Interest rate", "Vacancy rate", "adj_price", "adj_value"] #use 4 columns to predict 1 column #features selection
target = "change"
```

# **IV. Build model**

```php
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
#this will tell the accuracy percentage , when the model predicts the price would go up, how often did it go up and when the model predicts the price would go down how often did it go down
import numpy as np
```

**First I will create a prediction function**. This function will **take in** these inputs below and **return** predictions.
* a training set for model to train
* a test set which is what I want to make predictions on
* a set of predictors which are the columns I am going to use to make predictions
* a target show the house price goes up or down

**Then I create a backtest function**. This function will let me generate predictions for most of my data set but do it in a way that respects the order of the data set so that I can avoid using future data to predict the past.

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

Now, it shows that I have **59%** accuracy in my predictions.

# **V. Improve model**

Now, I will need to **add more variables** into my model to give the model extra information.

To do that, I will **add new columns** to price_data to show the ratio for each variable between its current data and its average data in last year in each row, using rolling method.

```php
yearly = price_data.rolling(52, min_periods=1).mean() #minimum periods =1 means that even if I have 1 period of data, pandas won't return a NaN. For example, pandas has less than 52 weeks of data (2008) from the current row backwards it'll return an Nan but min periods avoids that
yearly
```

![image](https://user-images.githubusercontent.com/131565330/235103623-1217239c-31ea-4b35-af17-e067eef43d11.png)

* Now, **the values in 'yearly'** are actually the average of Interest rate, average of Vacancy rate, average of CPI, average of Median Sale Price and average of	House Value, average of adj_price and average of adj_value for the past year.

* **For example,** in 'yearly', those figures in 2022-04-09 are actually the average of those in 2021.
Then I will find **the ratio** between the current value and the value in the past year to see how the price trending.

```php
yearly_ratios = [p + '_year' for p in predictors] #define columns names -> Interest rate_year, Vacancy rate_year,...
price_data[yearly_ratios] = price_data[predictors] / yearly[predictors] #add new columns and lấy data ở các column Interest rate,... trong df price_data (data hiện tại) chia cho data ở các column Interest rate trong df yearly (data past year) để ratio-> xem tăng hay giảm so với năm trước
price_data
```

![image](https://user-images.githubusercontent.com/131565330/235103941-1d0785b9-8f0a-432a-98c1-68075c335b4b.png)

**For example:** In 2022-04-09, Interest rate_year = 1.4 which mean the Interest rate in 2022-04-09 has increased **(Interest rate in 2022-04-09 = 1.4 x Interest rate in 2021-04-09)**.

I will check again if the whether accuracy improved or not with new variable **(yearly_ratios)**.

```php
preds, accuracy = backtest(price_data, predictors + yearly_ratios, target)
```

![image](https://user-images.githubusercontent.com/131565330/235104253-86bbf3d9-0bae-4dbe-8f37-c9a59e66b90e.png)

**Now, it shows that I have 65% accuracy in my predictions**. This means adding in these ratios has given the algorithm good information that it can use to make better decisions.

# **VI. Future value prediction**

After part V, I know that **adding more ratios will bring predictions with a higher accuracy rate** (65%). 

* Therefore, I will make a new dataframe called **price_data_need_to_predict** which is the same as original dataset (**price_data**) 

* But then I will add 4 new variables to **price_data_need_to_predict**

```php
yearly = price_data_need_to_predict.rolling(52, min_periods=1).mean() 
yearly_ratios = [p + '_year' for p in predictors] #define columns names -> Interest rate_year, Vacancy rate_year,...
price_data_need_to_predict[yearly_ratios] = price_data_need_to_predict[predictors] / yearly[predictors] #lấy data ở các column Interest rate,... trong df price_data (data hiện tại) chia cho data ở các column Interest rate trong df yearly (data past year) để ratio-> xem tăng hay giảm so với năm trước
price_data_need_to_predict
```

![image](https://user-images.githubusercontent.com/131565330/235104575-325b69f9-f292-43a0-9162-377e2e92dfb0.png)

Now, we can see that the new dataframe has **more variables than** the old one.

* **Old:** price_data (7 variables: Interest rate,	Vacancy rate,	CPI	Median Sale Price,	House Value,	adj_price,	adj_value)

* **New:** price_data_need_to_predict (11 variables= 7 old variables + 4 new variables: Interest rate_year, Vacancy rate_year, adj_price_year, adj_value_year)

**I will drop rows** in price_data_need_to_predict with null value in column 'next quarter' because next quarter of those days have not happenend yet (which means the future).

```php
new_price_data_need_to_predict = price_data_need_to_predict.copy() #make copy
new_save= new_price_data_need_to_predict.tail(13) #save data to predict later
price_data_need_to_predict.dropna(inplace=True)#drop na value
price_data_need_to_predict["change"] = (price_data_need_to_predict["next_quarter"] > price_data_need_to_predict["adj_price"]).astype(int) #add change as comparison between next_quarter and adj_price in reality
price_data_need_to_predict
```

![image](https://user-images.githubusercontent.com/131565330/235112838-33b4bfef-0f57-4cc6-ad23-17d7d1388ef1.png)

Now, I will have **8 predictors** because part V show that add those 4 variables would improve the chance of accuracy predictions.

```php
predictors = ['Interest rate', 'Vacancy rate', 'adj_price', 'adj_value', 'Interest rate_year'	,'Vacancy rate_year'	,'adj_price_year',	'adj_value_year']
def improved_predict(train, test, predictors, target):
    rf = RandomForestClassifier(min_samples_split=10, random_state=1) #min protects against overfitting by preventing the nodes in the decision trees in the random forest from splitting too deeply, a random state to ensure that every time I run my model it's going to generate the same sequence of random numbers, thereby giving the same result
    rf.fit(train[predictors], train[target]) #fit model using training data
    preds = rf.predict(test[predictors]) #generate predictions using test set
    return preds
def backtest(data, predictors, target):
    all_preds = [] #Create a list called all predictions
    for i in range(START, data.shape[0], STEP):
        train = price_data_need_to_predict.iloc[:i] #everything up until i
        test = price_data_need_to_predict.iloc[i:(i+STEP)] #the year following i
        all_preds.append(predict(train, test, predictors, target)) #append all of prediction sets to the list
    preds = np.concatenate(all_preds) #all predictions is going to be a list of numpy arrays (all_preds) -> concatenate those arrays together into a single array
    return preds, accuracy_score(data.iloc[START:][target], preds) # (data.iloc[START:][target]) are Actual values for test data vs (preds) Prediction values
```

Then, I have a new model which use **8 factors** to predict the 'Change'(trend of house price) with the function call improved_predict.

Now, I will run the model with: 
* **price_data_need_to_predict as training set:** because this new dataframe now includes 8 factors

* **new_save as test set:**  because this is the dataset contain rows that have null values of next quarter from 2022-04-16 to 2022-07-09 thereby it wil be use to predict future value after model is trained

* **K as future value:** this will be the prediction for next quarter from 2022-04-16 to 2022-07-09 (would be houses price trend from 2022-07-16 to 2022-10-09)
![image](https://user-images.githubusercontent.com/131565330/235105584-bcfaaace-311c-448c-80ae-2d60c0af1932.png)

```php
new_save['change'] = K
new_save #future data with predict
```
![image](https://user-images.githubusercontent.com/131565330/235113180-64c775a5-4f90-4d69-a4dc-89c1e2822eb2.png)

Now, the value in 'Change' column will show the **future trends of house prices**.

**For example:** 

**In 2022-04-16,** change = 0

* This means: the price of next quarter **(on this day 3 months later)** < adj price **(on this day)**

* Which also mean: adj price of **2022-07-16** (future) < adj price of **2022-04-16** (present)

**In 2022-06-25,** change = 1

* This means: the price of next quarter **(on this day 3 months later)** > adj price **(on this day)**

* Which also mean: adj price of **2022-09-25** (future) > adj price of **2022-06-25** (present)

**Finally,** I will test this model accuracy again to see if the idea of adding more variable could bring more percentage of accuracy as it shows before **in part V**.

![image](https://user-images.githubusercontent.com/131565330/235113313-22e7e1da-2a1a-4e50-880b-d7be2ba4c1db.png)

**In conclusion,** I have more than **65%** accurate for my future prediction from 2022-04-16 to 2022-07-09.

Now I will **visualize** the predictions in **price_data_need_to_predict**.

```php
pred_match = (preds == price_data_need_to_predict[target].iloc[START:])
pred_match[pred_match == True] = "green" #True if it predict 1 and the next quarter price actual > adj price and  or it predict 0 and the next quarter price actually < adj price 
pred_match[pred_match == False] = "red" #False if it predict 1 and the next quarter price actual < adj price and  or it predict 0 and the next quarter price actually > adj price 
import matplotlib.pyplot as plt
plot_data = price_data_need_to_predict.iloc[START:].copy()
plot_data.reset_index().plot.scatter(x="index", y="adj_price", color=pred_match)
```

![image](https://user-images.githubusercontent.com/131565330/235106202-460a4a86-f968-4a19-8e6a-5c914b967c7c.png)

**The graph above shows that** when the house price is in uptrend, the model tends to provide more accurate predictions.

# **VII. Further ideas to keep improving model** 

Adding in more predictors that **explain when the market is about to shift** like news articles or other economic data sets, economic indicators, stock market indicators, gold price, place information where new house is being built, criminal rate, etc could help improve this model.



   






