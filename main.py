import pandas as pd

data_set = pd.read_csv('2019-Oct.csv', nrows = 50)

# , nrows = 50 kısmını silersen tüm data setle çalışıyosun, göstermesi kolay olsun diye duruyor

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


data_set.dropna(subset=['category_code' , 'brand'], inplace=True) # NaN's


print("Number of rows in the dataset:", len(data_set))

