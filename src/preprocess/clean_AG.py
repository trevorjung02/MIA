with open('data/newsSpace', encoding='latin') as f:
    file = f.read()
file = file.replace("\t\\N", "\tnone")
file = file.replace("\\\n", " ")
with open('data/newsSpace', 'w', encoding='latin') as f:
    f.write(file)
