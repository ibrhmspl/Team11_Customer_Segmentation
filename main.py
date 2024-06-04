import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense




data_set = pd.read_csv('2019-Oct.csv', nrows=6000)

# Yalnızca "purchase" işlemlerini filtreleyelim
purchase_df = data_set[data_set['event_type'] == 'purchase']

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

from sklearn.metrics import silhouette_score, classification_report, f1_score, recall_score, precision_score, \
    accuracy_score

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

# Random Forest
data_set = data_set.dropna()
X = data_set[['product_id', 'category_id', 'price']]
y = data_set['user_id']
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)


y_pred = rf_classifier.predict(X_test)
print("RandomForest:",classification_report(y_test, y_pred, zero_division=1))
feature_importances = rf_classifier.feature_importances_
features = ['product_id', 'category_id', 'price']
plt.figure(figsize=(10, 6))
sns.barplot(x=features, y=feature_importances)
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)
print("Naive Bayes",classification_report(y_test, y_pred, zero_division=1))

feature_importances = rf_classifier.feature_importances_
features = ['product_id', 'category_id', 'price']
plt.figure(figsize=(10, 6))
sns.barplot(x=features, y=feature_importances)
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

#KNN

knn = KNeighborsClassifier(n_neighbors=5)


knn.fit(X_train, y_train)


y_pred = knn.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


accuracy_list = []

for k in range(1, 31):
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10)
    accuracy_list.append(scores.mean())


plt.plot(range(1, 31), accuracy_list)
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validated Accuracy')
plt.title('KNN Classifier Accuracy')
plt.show()


optimal_k = accuracy_list.index(max(accuracy_list)) + 1
print(f'Optimal number of neighbors: {optimal_k}')

# en uygun k değerini bul
knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
knn_optimal.fit(X_train, y_train)
y_pred_optimal = knn_optimal.predict(X_test)


accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
print(f'Optimal Accuracy: {accuracy_optimal}')

conf_matrix_optimal = confusion_matrix(y_test, y_pred_optimal)
print('Optimal Confusion Matrix:')
print(conf_matrix_optimal)

class_report_optimal = classification_report(y_test, y_pred_optimal)
print('Optimal Classification Report:')
print(class_report_optimal)

#Replicating Human Behaviour Faruk
users_with_purchase = data_set.groupby('user_id').filter(lambda x: 'purchase' in x['event_type'].values)

#get_user_data, brings the data of customers that has made a purchase

def get_user_data(user_id, data):
    #
    user_data = data[data['user_id'] == user_id]
    purchase_indices = user_data[user_data['event_type'] == 'purchase'].index
    # If that customer made a purchase; Otherwise this function will not be correctly executed resulted in error.

    interaction_data = pd.DataFrame()
    for idx in purchase_indices:
        interaction_data = pd.concat([interaction_data, user_data.loc[:idx]])
        #brings all of the instances of that UserID (Either View or Purchase doesn't matter if user made a single purchase)
    return interaction_data
#returns in the form of a dataframe with that customer instances

#train_user_model, For a single user trains the linear regression model and returns it for further use
def train_user_model(user_id, data):
    #the function above that brings the necessary data
    user_data = get_user_data(user_id, data)
    X = np.arange(len(user_data)).reshape(-1, 1)
    y = user_data['price'].values
    model = LinearRegression()
    model.fit(X, y)
    return model


#predict_next_price, Using the function above to train the model predicts the next possible price. According to a given price
def predict_next_price(user_id, current_price, data):
    model = train_user_model(user_id, data)
    last_interaction_index = len(data[data['user_id'] == user_id]) - 1
    next_interaction_index = last_interaction_index + 1
    predicted_price = model.predict(np.array([[next_interaction_index]]))
    return predicted_price

#Testing Replicating Human Behaviour
#Handpicked user that made a purchase.
user_id = 551377651
current_price = 642
predicted_price = predict_next_price(user_id, current_price, users_with_purchase)
range_ = (current_price - predicted_price)/10
min_ = predicted_price - range_
max_ = predicted_price + range_
print(f"Predicted next price for user {user_id}: {predicted_price}")
print(f"Next View Item Range for User {user_id}: {min_} to {max_}")
# Her kullanıcı için toplam alışveriş tutarını hesaplayalım
user_purchase_total = purchase_df.groupby('user_id')['price'].sum().reset_index()
user_purchase_total.columns = ['user_id', 'total_spent']

# Her kullanıcı için alışveriş yapma sıklığını (alışveriş sayısı) hesaplayalım
user_purchase_freq = purchase_df.groupby('user_id')['event_time'].count().reset_index()
user_purchase_freq.columns = ['user_id', 'purchase_count']

# Toplam tutar ve sıklık verilerini birleştirelim
user_stats = pd.merge(user_purchase_total, user_purchase_freq, on='user_id')

#########################################################################################################
# 4 cluster olan kmeans ile segmentasyon
kmeans = KMeans(n_clusters=4, random_state=42)#4 cluster
user_stats['segment'] = kmeans.fit_predict(user_stats[['total_spent', 'purchase_count']])#kmeans için parametreler


segment_summary = user_stats.groupby('segment')[['total_spent', 'purchase_count']].mean()#segment parametrlerinin ortalama degerleri
print("Segmentlerin toplam harcama ve tekrarlanan alışveriş ortalama değeri")
print(segment_summary)#ortalama degerleri yazdirma

# Her bir segmentteki kullanici sayisi
segment_counts = user_stats['segment'].value_counts()
print("Segmentlerdeki kullanıcı sayıları")
print(segment_counts)

# LSTM için veri hazırlama
user_series = []
data_set['event_time'] = pd.to_datetime(data_set['event_time'])

for user_id, user_data in purchase_df.groupby('user_id'):
    user_data = user_data.sort_values(by='event_time')
    user_data['time_diff'] = data_set['event_time'].diff().dt.total_seconds().fillna(0)
    user_data['cumulative_spent'] = user_data['price'].cumsum()

    # LSTM parametreleri
    user_series.append(user_data[['time_diff', 'cumulative_spent']].values)

max_len = max(len(series) for series in user_series)#en uzun zaman serisine gore digerleri de ayni uzunluga getiriliyor
user_series_padded = np.array([np.pad(series, ((0, max_len - len(series)), (0, 0)), 'constant') for series in user_series])
user_stats[['total_spent', 'purchase_count']] = scaler.fit_transform(user_stats[['total_spent', 'purchase_count']])

#modele verebilmek icin normalizasyon
X = user_stats[['total_spent', 'purchase_count']].values.reshape((-1, 1, 2))
y = user_stats['segment'].values.reshape((-1, 1))

model = Sequential()#LSTM modeli
model.add(LSTM(64, input_shape=(max_len, 2), return_sequences=True))#parametreler time_diff ve cumulative_spent
model.add(LSTM(32, return_sequences=False))
model.add(Dense(4, activation='softmax'))  # 4 segment olusturuyoruz
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

y = tf.keras.utils.to_categorical(y, 4)

model.fit(X, y, epochs=50, batch_size=32, validation_split=0.2)#egitim adımı

# Segment tahminleri yapalım
segments = model.predict(X)#test seti prediction
segment_labels = np.argmax(segments, axis=1)
true_labels = np.argmax(y, axis=1)



# onceki segment degerleri user_stats icinde segemnt columnunda tutuluyordu
#LSTM segment degerleri user_segments icinde tutuluyor
user_segments = pd.DataFrame({'user_id': purchase_df['user_id'].unique(), 'segment': segment_labels})

# Kullanıcıların segmentlerini gerekli sutuna ekleme
user_segments_df = user_segments.copy()  #
user_stats_subset = user_stats[['user_id', 'total_spent', 'purchase_count']]
user_segments_df = pd.merge(user_segments_df, user_stats_subset, on='user_id', how='left')

#KMeans icin gorsellestirme
# Her segmentin toplam harcama ve alışveriş sıklığı ortalamalarını hesaplayalım
segment_summary = user_stats.groupby('segment')[['total_spent', 'purchase_count']].mean().reset_index()

# Toplam harcama ve alışveriş sıklığına göre segmentleri görselleştirelim
plt.figure(figsize=(12, 6))
sns.barplot(x='segment', y='total_spent', data=segment_summary)
plt.title('Segmentlere Göre Ortalama Toplam Harcama')
plt.xlabel('Segment')
plt.ylabel('Ortalama Toplam Harcama')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='segment', y='purchase_count', data=segment_summary)
plt.title('Segmentlere Göre Ortalama Alışveriş Sıklığı')
plt.xlabel('Segment')
plt.ylabel('Ortalama Alışveriş Sıklığı')
plt.show()

# Her segmentteki kullanıcı sayısını görselleştirelim
segment_counts = user_segments['segment'].value_counts().reset_index()
segment_counts.columns = ['segment', 'user_count']

plt.figure(figsize=(12, 6))
sns.barplot(x='segment', y='user_count', data=segment_counts)
plt.title('Segmentlere Göre Kullanıcı Sayısı')
plt.xlabel('Segment')
plt.ylabel('Kullanıcı Sayısı')
plt.show()

# Metriklerin hesaplanması
accuracy = accuracy_score(true_labels, segment_labels)
precision = precision_score(true_labels, segment_labels, average='weighted')
recall = recall_score(true_labels, segment_labels, average='weighted')
f1 = f1_score(true_labels, segment_labels, average='weighted')

# Sonuc metrikleri
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')

#LSTM icin gorsellestirme
segment_summary = user_segments.groupby('segment')[['total_spent', 'purchase_count']].mean().reset_index()

# Toplam harcama ve alisveris sıkligina gore segmentler
plt.figure(figsize=(12, 6))
sns.barplot(x='segment', y='total_spent', data=segment_summary)
plt.title('Segmentlere Göre Ortalama Toplam Harcama')
plt.xlabel('Segment')
plt.ylabel('Ortalama Toplam Harcama')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='segment', y='purchase_count', data=segment_summary)
plt.title('Segmentlere Göre Ortalama Alışveriş Sıklığı')
plt.xlabel('Segment')
plt.ylabel('Ortalama Alışveriş Sıklığı')
plt.show()

# Her segmentteki kullanici sayisi
segment_counts = user_segments['segment'].value_counts().reset_index()
segment_counts.columns = ['segment', 'user_count']

plt.figure(figsize=(12, 6))
sns.barplot(x='segment', y='user_count', data=segment_counts)
plt.title('Segmentlere Göre Kullanıcı Sayısı')
plt.xlabel('Segment')
plt.ylabel('Kullanıcı Sayısı')
plt.show()
