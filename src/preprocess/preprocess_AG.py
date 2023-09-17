import pandas
import html
import unicodedata
path = 'data/newsSpace'

df = pandas.read_csv(path, names=['source', 'url', 'title', 'image', 'category', 'description', 'rank', 'pubdate', 'video'], sep='\t', encoding='utf-8', na_values=['none'], on_bad_lines='skip', usecols=['category', "description"])

df = df.query("description.str.len() > 10")
df = df[df['description'].astype(str).str.contains("<") == False]
df['description'] = df['description'].map(html.unescape)
df['description'] = df['description'].map(lambda x : unicodedata.normalize('NFKD', x))
df = df.sample(frac=1)

df_oracle = df[df['category'].isin({'World', 'Sports', 'Business', 'Sci/Tech'})]
df_other = df[df['category'].isin({'U.S.', 'Europe', 'Music Feeds', 'Health', 'Software and Developement', 'Entertainment'})]

dataset_size = 60000
df_oracle[:dataset_size].to_csv('data/newsSpace_oracle_target_train.csv', encoding='utf-8')
df_oracle[dataset_size:2*dataset_size].to_csv('data/newsSpace_oracle_ref_train.csv', encoding='utf-8')
df_oracle[2*dataset_size : 3*dataset_size].to_csv('data/newsSpace_oracle_val.csv', encoding='utf-8')

df_other[:dataset_size].to_csv('data/newsSpace_other_ref_train.csv', encoding='utf-8')