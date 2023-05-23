from collections import Counter

text = open('data/enwik8', 'r').read() 
char_counts = Counter(text)

aftertext = ''
for s in text:
    if s == ' ':
        continue
    if char_counts[s]<10000:
        continue
    aftertext+=s
char_counts = Counter(aftertext)
open('data/enwik8_ex', 'w').write(aftertext) 
print(char_counts)
print(len(list(char_counts.keys())))