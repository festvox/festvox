import os, sys
import glob

class Judith(object):

    def __init__(self):
       self.invocation = 1

    def get_latest(self, folder):

       files_path = os.path.join(folder, '*.pth')
       files = sorted(
                 glob.iglob(files_path), key=os.path.getctime, reverse=True) 
       return files[0]

    def act_upon_action(self, action, projects_dir):

       msg = action

       return {'fulfillmentText': msg}


