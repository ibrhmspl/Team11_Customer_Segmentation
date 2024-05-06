import pandas as pd
from matplotlib import pyplot as plt


#data_set = pd.read_csv('2019-Oct.csv')


# , nrows = 50 kısmını silersen tüm data setle çalışıyosun, göstermesi kolay olsun diye duruyor


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


#data_set.dropna(subset=['category_code' , 'brand'], inplace=True) # NaN's
#519107250
#550050854
#535871217
#514336739
#539194858
#544648245
#
#

#filtered_data = data_set[data_set['event_type'] == 'purchase']
#filtered_data.to_csv('purchase_events.csv',index =False)
#New data-set containing purchase events called purchase_event.csv


#print("Number of rows in the dataset:", len(filtered_data))

df = pd.read_csv('purchase_events.csv')

# Calculate the size of each chunk
chunk_size = len(df) // 4

# Create four parts
for i in range(4):
    start = i * chunk_size
    # If it's the last chunk, include the remainder
    end = start + chunk_size if i < 3 else None
    # Slice the DataFrame
    chunk = df.iloc[start:end]
    # Write to a new CSV file
    chunk.to_csv(f'purchase_events_part_{i+1}.csv', index=False)

print("Data has been divided into 4 parts.")

#data_set['event_type'].hist()
#plt.xlabel('price')
#plt.show()

#print(filtered_data)