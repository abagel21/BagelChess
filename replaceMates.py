import pandas as pd
df = pd.read_csv("util/compiled_standard_june_2020_mates.csv")
# replace all scores involving forced mates with +/- 10000
count = 0
def replaceForcedMate(x):
    global count
    print(count)
    count = count + 1
    if x[0] == "#" :
        if x[2] == '0' :
            if x[1] == "+" :
                return +10000
            else :
                return -10000
        else : 
            if x[1] == "+" :
                return +5000
            else:
                return -5000
    else:
        return x

df['stockfish_eval'] = df['stockfish_eval'].apply(lambda x : replaceForcedMate(x))
df.to_csv("compiled_standard_june_2020.csv", index=False)