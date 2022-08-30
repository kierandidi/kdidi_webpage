with open("2022-08-30-flowmodels1.md", 'w+') as f:
    count = 0
    for line in f:
        line.replace('$', '$$') 