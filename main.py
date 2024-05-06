import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import seaborn as sns

data_set = pd.read_csv('2019-Oct.csv', nrows=1000)



pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


data_set.dropna(subset=['category_code', 'brand'], inplace=True) # NaN's

grouped_sorted_data = data_set.sort_values(by='user_id').groupby('user_id')


for user_id, group_data in grouped_sorted_data:
    print(f"User ID: {user_id}")
    print(group_data.head())


features = ["price", "product_id"]


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_set[features])

# K-Means model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_scaled)


cluster_centers = kmeans.cluster_centers_
cluster_labels = kmeans.labels_

data_set["cluster"] = cluster_labels


# print("Number of rows in the dataset:", len(grouped_sorted_data))

plt.figure(figsize=(10, 7))
X = data_set[["price", "product_id"]]
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distances")
plt.show()


features = ["price"]


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_set[features])

# K-Means model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(data_scaled)


data_set["segment"] = kmeans.labels_


colors = {0: 'red', 1: 'blue', 2: 'green'}


plt.figure(figsize=(10, 6))
for segment, color in colors.items():
    segment_data = data_set[data_set['segment'] == segment]
    plt.scatter(segment_data['price'], segment_data['user_id'], color=color, label=f'Segment {segment}')
plt.xlabel('Price')
plt.ylabel('Customer ID')
plt.title('Customer Segmentation')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for segment, color in colors.items():
    segment_data = data_set[data_set['segment'] == segment]
    plt.hist(segment_data['price'], bins=20, color=color, alpha=0.5, label=f'Segment {segment}')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Customer Segmentation - Price Distribution')
plt.legend()
plt.show()

from sklearn.metrics import silhouette_score


silhouette_scores = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled)
    silhouette_scores.append(silhouette_score(data_scaled, kmeans.labels_))


plt.plot(range(2, 10), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Determining the Optimal Number of Clusters')
plt.show()








# Analyze the popularity of different product categories
category_popularity = data_set['category_code'].value_counts().head(10)

# Plot the results
plt.figure(figsize=(12, 6))
category_popularity.plot(kind='bar', color='skyblue')
plt.title('Top 10 Popular Product Categories')
plt.xlabel('Category')
plt.ylabel('Number of Events')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



#Müşteri sadakati analizi için gerekli verileri hazırla
customer_data = data_set.groupby('user_id').agg({'event_time': 'count', 'price': 'sum'}).reset_index()
customer_data.columns = ['user_id', 'total_transactions', 'total_spent']

# Tekrarlayan alışveriş yapma olasılığını hesapla
customer_data['repeat_probability'] = customer_data['total_transactions'] / customer_data['total_transactions'].sum()

# Analiz sonuçlarını görselleştir
plt.figure(figsize=(12, 6))
plt.hist(customer_data['repeat_probability'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Repeat Purchase Probability')
plt.ylabel('Number of Customers')
plt.title('Repeat Purchase Probability Distribution')
plt.show()

# Müşteri bazında toplam harcamaları hesapla
customer_spending = data_set.groupby('user_id')['price'].sum().reset_index()

# En çok harcama yapan müşteriyi bul
top_customer = customer_spending.loc[customer_spending['price'].idxmax()]

print("En çok harcama yapan müşteri ID:", top_customer['user_id'])
print("Toplam harcama miktarı:", top_customer['price'])

