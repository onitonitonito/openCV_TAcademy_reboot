"""
# Documentation configParser
# https://docs.python.org/3/library/configparser.html
"""
import configparser


# to load the user account information from config.ini(*hidden)
config = configparser.ConfigParser()
config.read('ghost_leg_config.ini')



if __name__ == '__main__':
    num_story = int(config.get('default', 'num_story'))

    names = [ name.strip().replace("'", '')
    for name in config.get('list', 'names').split('\n')
    if name != '']

    asks_keys = [ key.strip().replace("'", '')
    for key in config.get('dict', 'keys').split('\n')
    if key != '']

    asks_values = [ value.strip().replace("'", '')
    for value in config.get('dict', 'values').split('\n')
    if value != '']

    asks = {}
    for key, value in zip(asks_keys, asks_values):
        asks[key] = value

    print(type(num_story))
    print(num_story, '\n\n')

    print(type(names))
    print(names, '\n\n')

    print(type(asks))
    print(asks, '\n\n')
