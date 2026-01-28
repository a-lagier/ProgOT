def get_logger(out_file, **kwargs):
    return Logger(out_file, **kwargs)

def dict_to_str(dct, t=''):
    s = ''
    for k,v in dct.items():
        if isinstance(v, dict):
            s += t + k + ': \n' + dict_to_str(v, t=t+'  ')
        else:
            s += t + k + ': ' + str(v) + '\n'
    return s

class Logger():

    def __init__(self, out_file):

        self.out_file = out_file
        self.content = ''

    def log_config(self, cfg):
        self.content += dict_to_str(cfg)

    def log(self, *args, endline='\n'):
        self.content += ' '.join(str(a) for a in args) + endline
        print(' '.join(str(a) for a in args))

    def write(self):
        with open(self.out_file, 'w') as f:
            f.write(self.content)