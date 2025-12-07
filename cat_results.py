def read_results():
    try:
        with open(r'data/outputs/hawkes_results.txt', 'r') as f:
            print(f.read())
    except Exception as e:
        print(e)
read_results()
