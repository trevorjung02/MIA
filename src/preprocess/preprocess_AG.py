import pandas
import html
import unicodedata
path = 'data/newsSpace'

df = pandas.read_csv(path, names=['source', 'url', 'title', 'image', 'category', 'description', 'rank', 'pubdate', 'video'], sep='\t', encoding='utf-8', na_values=['none'], on_bad_lines='skip', usecols=["description"])  

df = df.query("description.str.len() > 10")
df = df[df['description'].astype(str).str.contains("<") == False]
df= df.applymap(html.unescape)
df = df.applymap(lambda x : unicodedata.normalize('NFKD', x))
df = df.sample(frac=1)

train_size = 60000
val_size = 60000
alt_size = 60000
df[:train_size].to_csv('data/newsSpace_train.csv', encoding='utf-8')
df[train_size:train_size+val_size].to_csv('data/newsSpace_val.csv', encoding='utf-8')
df[train_size+val_size : train_size+val_size+alt_size].to_csv('data/newsSpace_alt.csv', encoding='utf-8')