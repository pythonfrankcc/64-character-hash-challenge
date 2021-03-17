example = "Hello world"
a = example[::-1]
print(type(a))

#creating an additional column that gets the length of each hash to see whether there is a correlation with the results column in the future
df2['hash_length'] = df2['hash'].str.len()

