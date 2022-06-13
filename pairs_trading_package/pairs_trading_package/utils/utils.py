import hashlib
import datetime
import time
import random 
import string

def postfix_keys_to_dict(dict_obj, postfix):
    return {f'{k}_{postfix}': v for k, v in dict_obj.items()}

def get_current_time_hash():
  
    timestamp = datetime.datetime.now()

    unix_time = time.mktime(timestamp.timetuple())

    resulting_timestamp = str(unix_time)

    md5_digest = hashlib.md5((resulting_timestamp).encode('utf-8')).hexdigest()

    return md5_digest

def randomword(length):
    
    letters = string.ascii_lowercase
    
    return ''.join(random.choice(letters) for i in range(length))


def get_random_hash():
  
    md5_digest = hashlib.md5(randomword(50).encode('utf-8')).hexdigest()

    return md5_digest

