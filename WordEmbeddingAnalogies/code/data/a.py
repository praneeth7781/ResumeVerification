with open('Analogy_dataset.txt', 'r') as f:
    for line in f:
        a, b, c, d = line.split()
        print(a.lower(), b.lower(), c.lower(), d.lower())