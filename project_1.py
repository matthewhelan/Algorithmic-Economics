

import pandas as pd
import numpy as np
df = pd.read_csv ('scanner_data.csv')
#print (df)

print(  "Number of Customer_ID",    len(pd.unique(df['Customer_ID']))  )
print(  "Number of SKU  ",    len(pd.unique(df['SKU']))  )
date = pd.to_datetime(df['Date'], format='%d/%m/%Y')
week = [0]*len(date)
timestamp = [0]*len(date)

#print(df.isocalendar().week)

for i in range(len(date)):
  week[i] = (date[i].isocalendar()[1] + 1) % 54
  timestamp[i] = date[i].timestamp()
  
#print((week[index] + 1) % 54)
#print(date[index])
df['Week'] = week
df['Timestamp'] = timestamp

df = df.sort_values(by="Timestamp") #sorts
print(df.iloc[1:131705])


del week,date,timestamp
#we print out the table of the data imported and note the number of unique Customer_IDs and unique SKUs


#split table into weeks
#create 3d array, first sorted by Week number, then by product SKU and customer ID (order of last 2 doesn't matter)
#df = df[[]]
#df["Date"] = pd.to_datetime(df["Date"])
#df = df.sort_values(by="Date") #sorts
#print(df['Quantity'][i])
 
proxi = {} 
 
for i in range(len(df)):
  inner = (df.loc[i,'SKU'], df.loc[i,'Customer_ID'])
  q = df.loc[i,'Quantity']
  try:
    proxi[inner] += 1;
  except KeyError:
      proxi[inner] = 1;
  #print(inner, proxi[inner])
print("Proxi variable counting transactions:")
print(proxi)

#  if inner in proxi.keys():
#    proxi[inner] += 1
#  else:
#    proxi[inner] = 1

#How many unique SKUs?
skus = 0
s = {}

for k in range(len(df)):
  try:
    s[df.at[k,'SKU']] = s[df.at[k,'SKU']]
  except KeyError:
    s[df.at[k,'SKU']] = skus
    skus+=1

print("Unique SKUS:", skus)

#How many unique Customer IDs?

cust = 0
c = {}

for k in range(len(df)):
  try:
    c[df.at[k,'Customer_ID']] = c[df.at[k,'Customer_ID']]
  except KeyError:
    c[df.at[k,'Customer_ID']] = cust
    cust+=1

print("Unique Customer IDs:", cust)

#How many unique Weeks?

w = 0
weeks = {}

for k in range(len(df)):
  try:
    weeks[df.at[k,'Week']] = weeks[df.at[k,'Week']]
  except KeyError:
    weeks[df.at[k,'Week']] = w
    w+=1

print("Unique Weeks:", w)

"""Note that to simplify our runtime, we only run our analysis on the top most active customers. Here, we find a list of the top most active customers."""

#Determines a list of the 100 most active customers

quantity_list = ['Customer_ID', 'quantity']

quantity_df = pd.DataFrame(index=list(range(1, max(df['Customer_ID'])+1)), columns=quantity_list)
#print(quantity_list)

#defaults to empty, you can manually change by calling feature_df.at[CUSTOMER_ID, 'FEATURE_NAME'] = VALUE_TO_CHANGE_TO
#example:
#feature_df.at[1, 'Average Transaction Expense'] = 1


#sorts df by customer_id
df = df.sort_values(by="Customer_ID")
num_of_customers = max(df['Customer_ID'])

#we iterate this is needed
customer = 1 
total_items_for_specific_customer = 0
#print(df)

#    print("Customer ID", str(row["Customer_ID"]))

#Note that df is sorted by Customer_ID
#print(df)

for index, row in df.iterrows():
  if row["Customer_ID"] == customer:
    total_items_for_specific_customer = total_items_for_specific_customer + row["Quantity"] 
  #where we reach a new customer
  elif not row["Customer_ID"] == customer:
    quantity_df.at[customer, 'Customer_ID'] = customer
    quantity_df.at[customer, 'quantity'] = total_items_for_specific_customer


    #reset total_items_for_specific_customer
    total_items_for_specific_customer = row["Quantity"]
    customer = customer+1

#case for very lastmost customer
quantity_df.at[customer, 'Customer_ID'] = customer
quantity_df.at[customer, 'quantity'] = total_items_for_specific_customer

#now we sort quantity_df by quantity to get the 1000-most active customers
quantity_df = quantity_df.sort_values(by = 'quantity', ascending=False)

#print(quantity_df)

#now, we can get a list of the 1000 most active customers
top_cust_num = 100
top_100_list = []
count = 1
for index, row in quantity_df.iterrows():
  top_100_list.append(row['Customer_ID'])
  count = count + 1
  if count == (top_cust_num + 1):
    break
print("Top 100 most active customers: ",top_100_list)
print("Number of top customers in list: ", len(top_100_list))

"""We then parse our dataframe of transactions so that only transactions involving the people in our most active list remain. """

# Parse the data frame so that only transactions from the top 1000 active customers remain
#list of indexes to drop
labels_to_drop = []
labels_dropped = 0
for index, row in df.iterrows():
  if not row['Customer_ID'] in top_100_list:
    labels_to_drop.append(index)
    labels_dropped += 1
df=df.drop(labels=labels_to_drop, axis=0)
print(df)
print(labels_dropped, " transactions removed")


df = df.sort_values("Week")
df_copy = df
#print(df.dtypes)
#print(df)
Nan = float('NaN')

#This is just testing of how to make a new row and add it to the df dataframe
#df2 = pd.DataFrame({'Unnamed: 0':[Nan],'Date':[Nan], 'Customer_ID': [0000], 'Transaction_ID':[Nan] ,'SKU_Category':[Nan],'SKU':['not an option'], 'Quantity':[Nan], 'Sales_Amount':[0.00], 'Week':[week], 'Timestamp': [Nan]})
#print(df2)
#df = pd.concat([df, df2], ignore_index = True, axis = 0)
#print(df)


#we have a list of all 1000 customers
#print(df)
number_of_dictionaries_prime = 100000
lst_prime = [dict() for number in range(number_of_dictionaries_prime)]
count_prime = 0
top_100_list_copy = top_100_list.copy()
#print(top_100_list_copy)
week = 0
#We iterate through each transaction and mark off if that customer made a transaction from our list
for index, row in df.iterrows():
  #if we are in the same week, we check if the customer_id is in the list or not
  if row["Week"] == week:
      #remove customer_id if exists in list
      if row["Customer_ID"] in top_100_list_copy:
        top_100_list_copy.remove(row["Customer_ID"])
      #otherwise, we do nothing and move on
  
  #we check if we have reached our final index of the original column. If so, we finalize our list for this last week

  #This means we have reached a new week. Thus, we finalize our list for the previous week
  elif not row["Week"] == week:
    #for every remaining customer_ID, appends a new row to df representing an outside option
    for i in top_100_list_copy:
      lst_prime[count_prime] = {'Unnamed: 0':Nan,'Date':Nan, 'Customer_ID': i, 'Transaction_ID':0 ,'SKU_Category':Nan,'SKU':'not an option', 'Quantity':0.0, 'Sales_Amount':0.00, 'Week':week, 'Timestamp': Nan}
      count_prime += 1
      #df2 = pd.DataFrame({'Unnamed: 0':[Nan],'Date':[Nan], 'Customer_ID': [i], 'Transaction_ID':[Nan] ,'SKU_Category':[Nan],'SKU':['not an option'], 'Quantity':[Nan], 'Sales_Amount':[0.00], 'Week':[week], 'Timestamp': [Nan]})
      #df = pd.concat([df, df2], ignore_index = True, axis = 0)

    #We reset the Customer_ID_list_weekly and add +1 to the week
    top_100_list_copy = top_100_list.copy()
    week += 1
    if row["Customer_ID"] in top_100_list_copy:
        top_100_list_copy.remove(row["Customer_ID"])

#if it is the final week, we would reach here as our booleans don't catch it. So, we merely do the final week here
for i in top_100_list_copy:
  lst_prime[count_prime] = {'Unnamed: 0':Nan,'Date':Nan, 'Customer_ID': i, 'Transaction_ID':0 ,'SKU_Category':Nan,'SKU':'not an option', 'Quantity':0.0, 'Sales_Amount':0.00, 'Week':week, 'Timestamp': Nan}
  #df2 = pd.DataFrame({'Unnamed: 0':[Nan],'Date':[Nan], 'Customer_ID': [i], 'Transaction_ID':[Nan] ,'SKU_Category':[Nan],'SKU':['not an option'], 'Quantity':[Nan], 'Sales_Amount':[0.00], 'Week':[week], 'Timestamp': [Nan]})
  #df = pd.concat([df, df2], ignore_index = True, axis = 0)
del lst_prime[count_prime:]
df2 = pd.DataFrame(lst_prime, columns=df.columns.values.tolist())
#print(df2)
df = pd.concat([df,df2], ignore_index=True, axis=0)
print(df)
print(count_prime, " empty transactions added to the dataset")

"""Here, we split the transactions such that every transaction has quantity of 1. """

# number_of_dictionaries = 10000000
# lst = [dict() for number in range(number_of_dictionaries)]
# count = 0
# #we split up the multiple quantity orders into separate entries
# for index, row in df.iterrows():
#       #print(df.loc[index, 'Quantity'])
#       if (df.loc[index, 'Quantity'] > 1):
#         temp = df.loc[index, 'Quantity']
#         df.loc[index, 'Quantity'] = 1
#         for i in range(((int) (temp))-1):
#           d = {'Unnamed: 0':df.loc[index, 'Unnamed: 0'],
#                               'Date':df.loc[index, 'Date'], 
#                               'Customer_ID': df.loc[index, 'Customer_ID'], 
#                               'Transaction_ID':df.loc[index, 'Transaction_ID'] , 
#                               'SKU_Category': df.loc[index, 'SKU_Category'],
#                               'SKU':df.loc[index, 'SKU'], 'Quantity':1.0, 
#                               'Sales_Amount':df.loc[index, 'Sales_Amount']/temp, 
#                               'Week': df.loc[index, 'Week'], 
#                               'Timestamp': df.loc[index, 'Timestamp']}
          
#           lst[count] = d
#           #print(count)
#           count+=1
#         #if (count > 1000):
#         #  break
#           #df2 = pd.DataFrame(d)
#           #df = pd.concat([df, df2], ignore_index = True, axis = 0)

# del lst[count:]
# df2 = pd.DataFrame(lst, columns=df.columns.values.tolist())
# print(df2)
# df = pd.concat([df,df2], ignore_index=True, axis=0)

      #df2 = pd.DataFrame({'Unnamed: 0':[Nan],'Date':[Nan], 'Customer_ID': [i], 'Transaction_ID':[Nan] ,'SKU_Category':[Nan],'SKU':['not an option'], 'Quantity':[Nan], 'Sales_Amount':[0.00], 'Week':[week], 'Timestamp': [Nan]})
      #df = pd.concat([df, df2], ignore_index = True, axis = 0)
#print(df)

#NOTES: Possible features
#transaction history (times purchased previously for a given SKU)/Same with Category, average transaction price per SKU,


# Here, we are assigning each SKU a unique number value, where 'not an option' = 0

#construct your new variable
#df['feature 1'] = df.iloc[].sum(axis=1)

#CHANGE THIS FEATURE_LIST TO ADD/REMOVE FEATURES
#Features are defaulted to null unless changed
feature_list = ['Average Price Per Product', 'Total Quantity of Items Bought', 'Num of Unique Weeks Came', 'Unique Transactions', 'Average Weekly Frequency', 'Average Products Bought Per Visit','Average Expense Per Visit','Min Week', 'Max Week']

feature_df = pd.DataFrame(index=top_100_list, columns=feature_list)
#print(feature_df)
#defaults to empty, you can manually change by calling feature_df.at[CUSTOMER_ID, 'FEATURE_NAME'] = VALUE_TO_CHANGE_TO
#example:
#feature_df.at[1, 'Average Transaction Expense'] = 1


#sorts df by customer_id
df_copy = df_copy.sort_values(by="Customer_ID")
num_of_customers = max(df['Customer_ID'])

#we iterate this is needed
top_100_list.sort()
print(type(top_100_list))
print(top_100_list)
x = 0
customer = top_100_list[x]
total_spent_for_specific_customer = 0.00
total_items_for_specific_customer = 0
total_transactions_for_specific_customer = 0

unique_weeks_per_customer = []
unique_transaction_ids_per_customer = []

#for index, row in df.iterrows():
#    print("Customer ID", str(row["Customer_ID"]))

#Note that df is sorted by Customer_ID
print(df_copy)

for index, row in df_copy.iterrows():
  #print(index)
  #assume each transaction is of quantity 1...
  #print(row["Sales_Amount"])
  if row["Customer_ID"] == customer:
    #check if the week in that transaction is already in unique_weeks_per_customer
    if row["Week"] not in unique_weeks_per_customer:
      #if week is not already in, add it to the list
      unique_weeks_per_customer.append(row["Week"])

    #check if that transaction is UNIQUE - already in unique_transaction_ids_per_customer
    if row["Transaction_ID"] not in unique_transaction_ids_per_customer and not 0:
      #if transaction id is not in the list already, add it
      unique_transaction_ids_per_customer.append(row["Transaction_ID"])


    total_spent_for_specific_customer = total_spent_for_specific_customer + row["Sales_Amount"]
    #print(total_spent_for_specific_customer)
    total_items_for_specific_customer = total_items_for_specific_customer + row["Quantity"]

  if not row["Customer_ID"] == customer:
    num_of_unique_weeks_showed = len(unique_weeks_per_customer)
    feature_df.at[customer, 'Num of Unique Weeks Came'] = num_of_unique_weeks_showed
    feature_df.at[customer, 'Unique Transactions'] = len(unique_transaction_ids_per_customer)
    feature_df.at[customer,'Average Weekly Frequency'] = (len(unique_transaction_ids_per_customer)/num_of_unique_weeks_showed)
    feature_df.at[customer,'Min Week'] = min(unique_weeks_per_customer)
    feature_df.at[customer,'Max Week'] = max(unique_weeks_per_customer)


    #feature_df.at[customer,'Average Week'] = sum(unique_weeks_per_customer)/len(unique_weeks_per_customer)


    if not total_items_for_specific_customer == 0:
      avg_transaction_amount = total_spent_for_specific_customer/total_items_for_specific_customer
      #print(avg_transaction_amount)
      feature_df.at[customer, 'Average Price Per Product'] = avg_transaction_amount
      feature_df.at[customer, 'Total Quantity of Items Bought'] = total_items_for_specific_customer

    elif total_items_for_specific_customer == 0:
      feature_df.at[customer, 'Average Price Per Product'] = 0.00
      feature_df.at[customer, 'Total Quantity of Items Bought'] = total_items_for_specific_customer

    if len(unique_transaction_ids_per_customer) == 0:
        feature_df.at[customer, 'Average Products Bought Per Visit'] = 0
        feature_df.at[customer, 'Average Expense Per Visit'] = 0.00
    else:
        feature_df.at[customer, 'Average Products Bought Per Visit'] = total_items_for_specific_customer/len(unique_transaction_ids_per_customer)
        feature_df.at[customer, 'Average Expense Per Visit'] = total_spent_for_specific_customer/len(unique_transaction_ids_per_customer)
    
    #adds 1 to customer and resets total_spent_for_specific_customer
    x = x +1
    if x < 100:
      customer = top_100_list[x]
    total_spent_for_specific_customer = 0
    total_items_for_specific_customer = 0
    unique_weeks_per_customer = [row["Week"]]
    unique_transaction_ids_per_customer = []
    if row["Customer_ID"] == customer:
      total_spent_for_specific_customer = row["Sales_Amount"]
      total_items_for_specific_customer = 1

feature_df.at[customer, 'Num of Unique Weeks Came'] = len(unique_weeks_per_customer)
feature_df.at[customer, 'Unique Transactions'] = len(unique_transaction_ids_per_customer)
feature_df.at[customer,'Average Weekly Frequency'] = (len(unique_transaction_ids_per_customer)/num_of_unique_weeks_showed)
feature_df.at[customer,'Min Week'] = min(unique_weeks_per_customer)
feature_df.at[customer,'Max Week'] = max(unique_weeks_per_customer)


#feature_df.at[customer,'Average Week'] = sum(unique_weeks_per_customer)/len(unique_weeks_per_customer)
if total_items_for_specific_customer == 0:
    feature_df.at[customer, 'Average Price Per Product'] = 0.00
    feature_df.at[customer, 'Total Quantity of Items Bought'] = total_items_for_specific_customer
else:
  avg_transaction_amount = total_spent_for_specific_customer/total_items_for_specific_customer
  feature_df.at[customer, 'Average Price Per Product'] = avg_transaction_amount
  feature_df.at[customer, 'Total Quantity of Items Bought'] = total_items_for_specific_customer

if len(unique_transaction_ids_per_customer) == 0:
  feature_df.at[customer, 'Average Products Bought Per Visit'] = 0
  feature_df.at[customer, 'Average Expense Per Visit'] = 0.00
else:
  feature_df.at[customer, 'Average Products Bought Per Visit'] = total_items_for_specific_customer/len(unique_transaction_ids_per_customer)
  feature_df.at[customer, 'Average Expense Per Visit'] = total_spent_for_specific_customer/len(unique_transaction_ids_per_customer)

print(feature_df)

print(df['Customer_ID'])

#restucture your data

#df =  df[[]]

"""### Question 4
Produce the utility parameters $\beta_{0j}, \beta_{1j},\cdots \beta_{kj}$ and $\alpha_j$ for every product $j$  by estimating a multinomial 
logit model from your constructed dataset.
"""

#Generating Matrix X
all_columns = []
row_column = []
print(df)
#iterate through all transactions, find the corresponding feature set to the row['Customer_ID'], in addition to the row["Sales_Amount"]
for index, row in df.iterrows():
  for x in feature_df.columns:
    row_column.append(feature_df.at[row["Customer_ID"], x])
  if row["Sales_Amount"] == 0.00:
    row_column.append(0.00)
  else:
    row_column.append(-row["Sales_Amount"]/row["Quantity"])
  all_columns.append(row_column)
  row_column = []
X = np.array(all_columns)

#X = X[:500]

print(X.shape)

skus = 1
sku_to_index = {'not an option': 0}

for index, row in df.iterrows():
  try:
    sku_to_index[row["SKU"]] = sku_to_index[row["SKU"]]
  except KeyError:
    sku_to_index[row["SKU"]] = skus
    skus+=1
''' for k in range(len(df)):
  try:
    sku_to_index[df.at[k,'SKU']] = sku_to_index[df.at[k,'SKU']]
  except KeyError:
    print(k)
    sku_to_index[df.at[k,'SKU']] = skus
    skus+=1 '''
sku_to_price = {'not an option': 0}
for index, row in df.iterrows():
  if row["Quantity"] != 0.0:
    sku_to_price[row["SKU"]] = row["Sales_Amount"]/row["Quantity"]

index_to_sku = {v: k for k, v in sku_to_index.items()}
print("Unique SKUS:", skus)
print(index_to_sku)
print(len(sku_to_index))
print(sku_to_price)
print(len(df.SKU.unique()))

y = [0]*len(df)
for i in range(len(df)):
  y[i] = sku_to_index[df.at[i, 'SKU']]
print(y)

import sklearn
from sklearn.model_selection import train_test_split
X_train, X_test , y_train , y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.0000000000000000000000000000000000000000000000000000001, random_state = 42)

#Hint: you can use sklearn.linear_model.LogisticRegression() to achieve an estimation

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(multi_class = 'multinomial', penalty='none', solver= 'newton-cg', verbose = 1, max_iter = 1, fit_intercept = True, n_jobs=8, warm_start=True)
model.fit(X_train , y_train)

pred = model.predict(X_test)

print("The set of beta’s for the product 5: ",  model.coef_[5])
print(model.coef_)
print(model.coef_.shape)

score = model.score(X_test, y_test)
print(score)
#print(1/1606)

for i,p in zip(y_test, pred):
  print(i,":", p, "\tSKU", index_to_sku[p])

"""# Part 2
### Question 1
Construct a multi-armed bandit algorithm such that

1. It is randomly initialized at first and selects **one** product out of $j$ available products.
2. It updates  $\beta_{0j}, \beta_{1j},\cdots \beta_{kj}$ and $\alpha_j$  over  time by observing the utility $\widehat{u}_{ij}$ of each product $j$ it selected in the past and selects new products

"""

''' print(sku_df)
import random
random.seed(42)
#we need to select a random SKU
random_index = random.randint(0, skus)
random_sku = sku_df.at[random_index, 'sku']
print(random_sku) '''

#Hint: Try ridge regression on each arm separately,
import random
import math

e = .5
Beta_coef = np.ones(shape=model.coef_.shape)

def decide(customer_idx):
  explore = random.uniform(0,e)
  cust_feat = feature_df.iloc[customer_idx].to_numpy()#feature_df[top_100_list[customer_idx]]
  
  if explore < e:
    index = random.randint(0, len(model.coef_)-1)
  else:
    index = model.predict(cust_feat)
  sku = index_to_sku[index]
  Beta = model.coef_[index]
  cust_feat = np.append(cust_feat, sku_to_price[sku])
  utility = np.dot(cust_feat, Beta)
  if math.isnan(utility):
    print(cust_feat, ":", Beta, ":" ,utility)
  return sku, Beta, utility
''' cust_feat = feature_df.iloc[0].to_numpy()
sku = index_to_sku[0]
cust_feat = np.append(cust_feat, sku_to_price[sku])
print(sku_to_price[sku])
print(cust_feat) '''

def update_parameter(sku, Beta, utility, N):
  # keep track of number of times each product is selected
  # keep track of eman for each product
  index = sku_to_index[sku]
  Beta_coef[index] = ((N+1)*Beta_coef[index] - Beta_coef[index] + Beta)/(N+1)
  return

"""### Question 2

 Draw 1000 random consumers from your data. For each consumer,  run your online learning algorithm for 100 steps. Note that this is a simulation process --- i.e., your algorithm itself does not know $\beta_{0j}, \beta_{1j},\cdots \beta_{kj}$ and $\alpha_j$, but can only observe the $\widehat{u}_{ij}$ for any product $j$ that the algorithm pulled (i.e., purchased).     
 For each randomly picked consumer $i$, compute the difference $\Delta_i$ between the  maximum utility $\max_j\widehat{u}_{ij}$ (i.e., consumer $i$'s  utility for her  favorite product) and the average utility that your algorithm
achieved at the 100th step. Compute the average of $\Delta_i$ over those 1000 consumers, and explain why there is such a difference.  
"""

def rewards_difference(max_utility, average_utility):
    return max_utility - average_utility

def simulation():
  total_delta_i = 0
  for i in range(1000):
    rand_cust = random.randint(0, top_cust_num-1)
    max_utility = 0
    average_utility = 0
    for k in range(100):
      rand_sku, rand_Beta, current_utility = decide(rand_cust)
      if max_utility < current_utility:
        max_utility = current_utility
      average_utility += current_utility
      update_parameter(rand_sku, rand_Beta, current_utility, k)
    average_utility /= 100
    delta_i = rewards_difference(max_utility, current_utility)
    total_delta_i += delta_i
  return total_delta_i / 1000

print("Avg delta i = ", simulation())
print("Predicted Beta: ", Beta_coef)
print("Actual Beta: ", model.coef_)

"""Explain why there is such a difference.

**Please input your answer in this cell:**

The avg maximum utility is achieved by the customer gaining utility from his/her favorite item to purchase (averaged per iteration). Thus, the avg maximum utility is achievable if and only if the customer purchases only his or her favorite item, as otherwise, it would be impossible to maximize utility.

The average utility our algorithm achieved relies upon selecting a random product and doing either of two steps, the exploit step or the explore step (with probabilities 1-epsilon and epsilon respectively). During the exploit step, we choose the known item with the highest utility to the customer - note that this is only of the known items sets for the customer. During the explore step, we test the customer buying a randomized, other item. 

As a result, our algorithm relies on exploring unknown items (and exploiting, most likely, the items that are not the consumer's favorite) in order to find the avg utility provided at the 100th step. Meanwhile, the maximum avg utility is only possible if the consumer buys his/her favorite item (assuming constant utility provided per each purchase) for all iterations. Our algorithm requires purchasing other items, starting with a randomized SKU, meaning that we are guarenteeed to have a difference between the max avg utility and the avg utility calculated by our algorithm at the 100th step.
"""

