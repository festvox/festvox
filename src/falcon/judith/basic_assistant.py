# import flask dependencies
from flask import Flask, request, make_response, jsonify

import os, sys
from judith import Judith

import config

judith = Judith()

projects_dir = config.projects_dir

# initialize the flask app
app = Flask(__name__)

# default route
@app.route('/')
def index():
    return 'Hello World!'

# function for responses
def results():
    # build a request object
    req = request.get_json(force=True)

    # fetch action from json
    action = req.get('queryResult').get('action')

    # Read the file that contains information about the projects being tracked
    try:
      return judith.act_upon_action(action, projects_dir)
    except Exception as e:
      print(e)
      sys.exit() 

# create a route for webhook
@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    # return response
    return make_response(jsonify(results()))

# run the app
if __name__ == '__main__':
   app.run()


