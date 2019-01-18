from util import numpy_to_builtin, format_json, load_json, save_json, Path
from config import Config

class Experiment(object):
    def __init__(self, res, data, model):
        self.res = Path(res).mk()
        self.name = self.res._name
        self.data = Path(data)
        self.model = Path(model)
        self.flags = {}
        self.vars = {}

    def __repr__(self):
        return 'name=%s\nresult=%s\ndata=%s\nmodel=%s\nflags=%s\nvars=%s' % (self.name, self.res, self.data, self.model, self.flags, format_json(self.vars))
    
    def get(self, attr, default=None):
        return getattr(self, attr, default)
    
    @property
    def path(self):
        return self.res / 'experiment.json'

    @classmethod
    def from_res(cls, res):
        return cls.from_path(res / 'experiment.json')

    @classmethod
    def from_path(cls, path):
        if Path(path).exists():
            info = load_json(path)
            exp = Experiment(*[info[k] for k in ['result', 'data', 'model']])
            exp.flag(**info['flags'])
            exp.var(**info['vars'])
            return exp
        return None

    def save(self, f=False):
        self.res.mk()
        if not f and self.path.exists():
            print('Not saving experiment %s, already exists and "f" is not specified' % self.path)
        else:
            save_json(self.path,
                dict(result=self.res, data=self.data, model=self.model, flags=self.flags, vars=self.vars))
        return self
    
    def flag(self, *args, **kwargs):
        for a in args:
            self.flags[a] = []
        self.flags.update(kwargs)
        return self
    
    def unflag(self, *args):
        for a in args:
            del self.flags[a]
        return self
    
    def get_flags(self):
        flags = []
        for k, v in self.flags.items():
            k = k if k.startswith('-') else ('-' + k)
            if type(v) == list:
                flags.extend([k] + v)
            else:
                flags.extend([k, v])
        return list(map(str, flags))
    
    def unflag(self, *args):
        if len(args) == 0:
            self.flags = {}
        for a in args:
            del self.flags[a]
        return self

    def var(self, *args, **kwargs):
        for a in args:
            self.vars[a] = True
        self.vars.update(kwargs)
        return self
    
    def unvar(self, *args):
        if len(args) == 0:
            self.vars = {}
        for a in args:
            del self.vars[a]
        return self
    
    def set_vars(self):
        for k, v in self.vars.items():
            setattr(self, k, v)
        return self

    def cmd_full(self, gpu=None):
        cmd = 'python3 $LE/main.py -f %s -d %s -r %s' % (self.model, self.data, self.res)
        cmd = ' '.join([cmd] + self.get_flags())
        if gpu is not None:
            cmd = 'CUDA_VISIBLE_DEVICES=%s ' % gpu + cmd
        return cmd
    
    def cmd(self, gpu=None):
        cmd = 'python3 $LE/main.py -xp %s' % (self.path)
        if gpu is not None:
            cmd = 'CUDA_VISIBLE_DEVICES=%s ' % gpu + cmd
        return cmd


    def config(self, name=None, params={}):
        res = self.res / (name or ','.join(sorted('%s=%s' % kv for kv in params.items())))
        return Config(res, params)

    @property
    def best(self):
        return self.res / 'best_config'
    
    def link_best(self, name):
        self.best.link(name)

    def config_best(self):
        if self.best.exists():
            return self.config(self.best._real._name)
        return None
    

    @property
    def hyperband_state(self):
        return self.res / 'hyperband_state.json'

    def load_hyperband_state(self):
        if self.hyperband_state.exists():
            return { int(k): v for k, v in load_json(self.hyperband_state).items() }
        return None

    def save_hyperband_state(self, state):
        save_json(self.hyperband_state, numpy_to_builtin(state))
