import os, sys
import glob
import config

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


    def set_default_project(self, parameters):

       project_id = parameters['project'].lower()
       print("Checking for the project id ", project_id)
       assert os.path.exists(projects_dir + '/' + project_id)
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


    def act_upon_action(self, action, parameters=None):

       if action == 'set_default_project':
          return self.set_default_project(parameters)

       elif action == 'get_default_project':
          return self.get_default_project()

       else:
          msg = " I am sorry. I dont understand this " + action
          return {'fulfillmentText': msg}






