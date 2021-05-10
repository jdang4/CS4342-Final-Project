import pandas as pd 

# skip = [i for i in range(1, 6, 1)]
# df = pd.read_csv('test_submission2.csv', skiprows=skip, nrows=2)

# print(df.head())
# input()


# with open('test_submission2.csv') as fp:
#     count = 0
#     for _ in fp:
#         count += 1

# print(count)

df = pd.read_csv('submission.csv')
df2 = pd.read_csv('submission2.csv')

print(df.shape)
print(df2.shape)

df = df[['MachineIdentifier']].copy()
df2 = df2[['MachineIdentifier']].copy()

print(df.equals(df2))
