import pandas as pd

df = pd.read_csv('test.csv')

all_star_names = ['Dwight Howard',
'LeBron James',
'Jason Kidd',
'Dwyane Wade',
'Chris Bosh',
'Caron Butler',
'Chauncey Billups',
'Richard Hamilton',
'Kevin Garnett',
'Joe Johnson',
'Paul Pierce',
'Antawn Jamison',
'Tim Duncan',
'Carmelo Anthony',
'Allen Iverson',
'Yao Ming',
'Kobe Bryant',
'Brandon Roy',
'Chris Paul',
'Dirk Nowitzki',
'Amar\'e Stoudemire',
'Steve Nash',
'Carlos Boozer',
'David West']

df['AllStar'] = 0
df.loc[(df['Player'].isin(all_star_names)), 'AllStar'] = 1

df.to_csv('per_game_stats.csv', index=False)
