#!usr/bin/env python
# Program to mine data from your own facebook account

import json
import facebook
import os
import sys
import random

token = os.environ.get('FACEBOOK_TOKEN_JUDITH')
group_id = str(os.environ.get('DeEntanglement_GROUP_ID'))
research_dir = os.environ.get('research_dir')
quotes_file = research_dir + '/quotes.all'
print("The group is ", group_id)


def get_content(quotes_file):
    f = open(quotes_file)
    lines = []
    for line in f:
        line = line.split('\n')[0].split()
        content = ' '.join(k for k in line)
        if '/' in content:
           content = content.split('/')[0]
        if len(line) > 7:
           lines.append(content)
    selected_content = random.choice(lines).split()
    content = ' '.join(k for k in selected_content)
    text = "Do you remember where this is from ? " + '\n' + content 
    return text


def main():
    graph = facebook.GraphAPI(token)
    # profile = graph.get_object(
    #    'me', fields='first_name,location,link,email,groups')
    group = graph.get_object(id=group_id)
    id = group['id']

    #pic = get_pic()
    #pic = pics_path + '/' + pic
    #graph.put_photo(album_path=id + '/photos', image=open(pic, 'rb'), message='Look at this! Posting at ' + timestamp + ' EST')

    content = get_content(quotes_file)
    graph.put_object(id, 'feed', message=content)

    print(group)


if __name__ == '__main__':
    main()
