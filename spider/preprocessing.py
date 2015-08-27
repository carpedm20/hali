import json

j = json.loads(open('out_asked.json').read())
print(len(j))
#j = [i for i in j if len(i['q']) > 10]
j = [i for i in j if len(i['q'].split()) > 2]
print(len(j))
