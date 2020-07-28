import os, sys
import glob
import json
import neptune

projects_dir = 'sandbox'

class RemoteTracker(object):

    def __init__(self, exp_name, projects_dir='sandbox'):
       self.tracker = neptune
       self.tracker.init('srallaba/' + projects_dir)
       self.tracker.create_experiment(name=exp_name,
                              abort_callback=lambda: run_shutdown_logic_and_exit())
       print("Initialized Tracker") 

    def log_scalar(self, name, val):
       self.tracker.log_metric(name, val)


