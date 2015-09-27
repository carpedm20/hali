import requests
import json

headers = {
    'uid': 'ios-964197D5-5CF8-4753-82D0-DAEC4CD939E5',
    'Content-Type': 'application/json',
    'User-Agent': 'around/1.4.0 (iPhone; iOS 8.3; Scale/2.00)',
}

data = {'uid': 'ios-964197D5-5CF8-4753-82D0-DAEC4CD939E5'}

# Start

# users


r = requests.get('https://around.conbus.net/v2/users', headers={'uid': 'ios-812397D3-5CF1-2313-24A0-DBDC2CD919E2'})
j = json.loads(r.text)

r = requests.get('https://around.conbus.net/v2/articles?excludes=&size=30', headers=headers)
j = json.loads(r.text)

r = requests.post('https://around.conbus.net/v2/users/me/session/start', data=str(data).replace("'",'"'), headers=headers)
print r.text

uid = "ios-964197D5-5CF8-4753-82D0-DAEC4CD939E5"
new_data ={
    "birth": 1900,
    "gender": "NONE",
    "platform": "IOS",
    "uid": uid
}

r = requests.post('https://around.conbus.net/v2/users', data=str(new_data).replace("'",'"'), headers=headers)
print r.text

r = requests.get('https://around.conbus.net/v2/articles/searches/latest?page=0&size=1&type=Hashtag&value=%EB%B0%94%EB%B3%B4', headers=headers)
j = json.loads(r.text)
#print len(j['items'])
print j['items'][0]['content']

r = requests.get('https://around.conbus.net/v2/articles/searches/latest?page=1&size=20&type=Article&value=%EB%B0%94%EB%B3%B4', headers=headers)
j = json.loads(r.text)
print len(j['items'])

r = requests.get('https://around.conbus.net/v2/hashtags?keyword=+', headers=headers)
j = json.loads(r.text)
print j

r = requests.get('https://around.conbus.net/v2/articles?excludes=749&size=10', headers=headers)
j = json.loads(r.text)
with open('backup_around.json','w') as f:
    json.dump(j, f)
print len(j['items'])

MAX = 2288749
for i in xrange(MAX/3000):


r = requests.get('https://around.conbus.net/v2/articles?excludes=2288749%2C2285905%2C2239382%2C429958%2C2281912%2C2288755%2C2212418%2C2267640%2C2221967%2C2280588%2C1352615%2C2205148%2C1132393%2C2202756%2C2288709%2C2288797%2C2288736%2C2189018%2C2288726%2C2288735%2C483652%2C2201095%2C2288546%2C2204967%2C465871%2C2190557%2C2288603%2C2206344%2C2288364%2C2188424%2C2280891%2C2193240%2C2288519%2C2171007%2C2185920&size=30', headers=headers)
j = json.loads(r.text)
print len(j['items'])

# https://around.conbus.net/v2/articles/2031158/
