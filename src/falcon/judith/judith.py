import os, sys
import glob
import config
import json

projects_dir = config.projects_dir

class Judith(object):

    def __init__(self):
       self.invocation = 1
       self.default_project = None
       self.projects_dir = projects_dir

    def get_latest(self, folder):

       files_path = os.path.join(folder, '*.pth')
       files = sorted(
                 glob.iglob(files_path), key=os.path.getctime, reverse=True) 
       return files[0]

    def accelerate(self):

       with open(self.projects_dir + '/' + self.default_project + '/defaults.json') as json_file:
            info = json.load(json_file)
       slave = info['slave']
       acceleration_file = config.acceleration_file
       print("Boosting now")
       cmd = 'scp ' + acceleration_file + ' ' + slave + ':'
       print(cmd)
       os.system(cmd)

    def get_project_info(self, parameters):

       query = parameters['project_info'].lower()
       print("Checking for ", query)

       if query == 'acceleration':
          self.accelerate()
          msg = "Done Boss"
          return {'fulfillmentText': msg}


       with open(self.projects_dir + '/' + self.default_project + '/defaults.json') as json_file:
            info = json.load(json_file)
       val = info[query]
       msg = "The " + query + " is " + val
       return {'fulfillmentText': msg}

    def set_default_project(self, parameters):

       project_id = parameters['project'].lower()
       print("Checking for the project id ", project_id)

       try:
         assert os.path.exists(projects_dir + '/' + project_id)
       except AssertionError:
         msg = " I didnt catch that Boss. All I got was this " + project_id + " and it doesnt exist. Could you please repeat that"
         return {'fulfillmentText': msg}

       self.default_project = project_id

       msg = " I have set the default project to " + project_id
       return {'fulfillmentText': msg}


    def get_default_project(self):

       assert os.path.exists(projects_dir)
       default_project = self.default_project
 
       if default_project is not None:
          msg = " The current default project is " + default_project
       else:
          msg = " There is currently no default project set"

       return {'fulfillmentText': msg}

    def get_official_baseline(self):

       project = self.default_project
       assert project is not None 

    def post_to_facebook(self):
      
       

    def act_upon_action(self, action, parameters=None):

       if action == 'set_default_project':
          return self.set_default_project(parameters)

       elif action == 'get_default_project':
          return self.get_default_project()

       elif action == 'post_to_facebook':
          return self.post_to_facebook()

       elif action == 'project_info':

          try:
             assert self.default_project is not None
             return self.get_project_info(parameters)

          except AssertionError:
             msg = "Boss I dont have a default project"
             return {'fulfillmentText': msg} 

       else:
          msg = " I am sorry. I dont understand this " + action
          return {'fulfillmentText': msg}


    



