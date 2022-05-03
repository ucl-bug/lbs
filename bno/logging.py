import json

import neptune.new as neptune


def load_secrets():
    with open('secrets.json') as f:
        secrets = json.load(f)
    return secrets

class Logger(object):

  def __init__(self, name):
    self.name = name
    self.experiment = neptune.init(
      api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ZTFiMzA3OS02MjFmLTRiN2YtOTU5Ny1lZjE5MTk5YzgwNTcifQ==",
      project="astanziola/BNO",
      name=name,
    )
