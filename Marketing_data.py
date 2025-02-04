import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
pd.set_option("display.width",None)


creditcard_df=pd.read_csv("Marketing_data.csv")
print(creditcard_df.head())
print(creditcard_df.info())
print(creditcard_df.describe())

# Mean balance is $1564
# Balance frequency is frequently updated on average ~0.9
# Purchases average is $1000
# one off purchase average is ~$600
# Average purchases frequency is around 0.5
# average ONEOFF_PURCHASES_FREQUENCY, PURCHASES_INSTALLMENTS_FREQUENCY, and CASH_ADVANCE_FREQUENCY are generally low
# Average credit limit ~ 4500
# Percent of full payment is 15%
# Average tenure is 11 years

"""To know your customer to know which kind of product is tailored to them"""
# Let's see who made one off purchase of $40761

max_purchased_customer=creditcard_df[creditcard_df["ONEOFF_PURCHASES"]==40761.250000]
print(max_purchased_customer)

# Let's see who made cash advance of $47137

max_cash_advance_customer=creditcard_df[creditcard_df["CASH_ADVANCE"]==47137.21176]
print(max_cash_advance_customer)

"""Filling nulls using SIMPLE IMPUTER"""
print(creditcard_df.isnull().sum())

creditcard_df[["CREDIT_LIMIT","MINIMUM_PAYMENTS"]]=SimpleImputer(strategy="mean").fit_transform(creditcard_df[["CREDIT_LIMIT","MINIMUM_PAYMENTS"]])
print(creditcard_df.isnull().sum())

"""Seems CUST_ID(object) feature is irrelevant, so dropping it"""

creditcard_df.drop("CUST_ID",axis=1,inplace=True)
print(creditcard_df.info())

print(creditcard_df.duplicated().sum())

print(len(creditcard_df.columns))

"""Visualizing"""
# distplot combines the matplotlib.hist function with seaborn kdeplot()
# KDE Plot represents the Kernel Density Estimate
# KDE is used for visualizing the Probability Density of a continuous variable.
# KDE demonstrates the probability density at different values in a continuous variable.


plt.figure(figsize=(15,60))
for i in range(len(creditcard_df.columns)):
  plt.subplot(6, 3, i+1)
  sns.distplot(creditcard_df[creditcard_df.columns[i]], kde_kws={"color": "b", "lw": 3,}, hist_kws={"color": "g"})
plt.tight_layout()
plt.show()
plt.close()

correlation=creditcard_df.corr()
print(correlation)

plt.figure(figsize=(10,20))
sns.heatmap(correlation,annot=True)
plt.tight_layout()
plt.yticks(fontsize=5)
plt.xticks(rotation=45,fontsize=5)
plt.show()
plt.close()

# 'PURCHASES' have high correlation between one-off purchases, 'installment purchases, purchase transactions, credit limit and payments.
# Strong Positive Correlation between 'PURCHASES_FREQUENCY' and 'PURCHASES_INSTALLMENT_FREQUENCY'

"""Scaling the data to get better result"""
scaler=StandardScaler()
creditcard_df_scaled=scaler.fit_transform(creditcard_df)
print(creditcard_df_scaled)


"""Using Elbow method to find optimal number of clusters"""

scores_1=[]

range_values=range(1,20)
for i in range_values:
  kmeans=KMeans(n_clusters=i)
  kmeans.fit(creditcard_df_scaled)
  scores_1.append(kmeans.inertia_)

plt.plot(scores_1,"bx-")
plt.xlabel("Clusters")
plt.ylabel("WCSS")
plt.title("Finding optimal number of clusters")
plt.show()
plt.close()

# From this we can observe that, 4th cluster seems to be forming the elbow of the curve.
# However, the values does not reduce linearly until 8th cluster.
# Let's choose the number of clusters to be 7.

kmeans=KMeans(8)
kmeans.fit(creditcard_df_scaled)
labels=kmeans.labels_
print(labels)

print(kmeans.cluster_centers_.shape)

cluster_centers = pd.DataFrame(data = kmeans.cluster_centers_, columns = [creditcard_df.columns])
print(cluster_centers)

# In order to understand what these numbers mean, let's perform inverse transformation
cluster_centers = scaler.inverse_transform(cluster_centers)
cluster_centers = pd.DataFrame(data = cluster_centers, columns = [creditcard_df.columns])
print(cluster_centers)


"""Assigning a Cluster column to our dataset and finding which cluster group the rows are belongs to"""

creditcard_df_cluster=pd.concat([creditcard_df,pd.DataFrame({"clusters":labels})],axis=1)
print(creditcard_df_cluster.head())

"""Plotting the histogram of various clusters"""
for i in creditcard_df.columns:
  plt.figure(figsize=(35, 5))
  for j in range(8):
    plt.subplot(1, 8,j+1)
    cluster = creditcard_df_cluster[creditcard_df_cluster['clusters'] == j]
    cluster[i].hist(bins=20)
    plt.title('{}    \nCluster {} '.format(i, j))

  plt.show()
  plt.close()



"""We are gonna classifying customers into"""
# First Customers cluster (Transactors): Those are customers who pay least amount of intrerest charges and careful with their money, Cluster with lowest balance ($104) and cash advance ($303), Percentage of full payment = 23%
# Second customers cluster (revolvers) who use credit card as a loan (most lucrative sector): highest balance ($5000) and cash advance (~$5000), low purchase frequency, high cash advance frequency (0.5), high cash advance transactions (16) and low percentage of full payment (3%)
# Third customer cluster (VIP/Prime): high credit limit $16K and highest percentage of full payment, target for increase credit limit and increase spending habits
# Fourth customer cluster (low tenure): these are customers with low tenure (7 years), low balance

# Transactors: Not very profitable for banks. Possible strategies: Encourage them to use more card benefits or offer rewards for carrying balances.
# Revolvers: Highly profitable due to interest charges. Possible strategies: Offer personalized loans or financial planning services.
# VIP Customers: Good credit, responsible spending. Possible strategies: Offer premium services, higher limits, and exclusive rewards to retain them.
# Low Tenure Customers: New customers with low engagement. Possible strategies: Encourage more spending through welcome bonuses, cashback, or promotional offers.


