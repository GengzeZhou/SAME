
class MetaAgent(type):
    registry = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if 'name' in attrs:
            MetaAgent.registry[attrs['name']] = cls

class BaseAgent(metaclass=MetaAgent):
    ''' Base class for an agent to generate and save trajectories. '''

    def __init__(self, args, envs, device=None):
        self.args = args
        self.envs = envs
        self.device = device
        self.results = {}
        self.logs = {}

        # Initialize the agent model
        self._build_model()
    
    def _build_model(self):
        raise NotImplementedError

    @staticmethod
    def get_agent(name):
        return globals()[name+"Agent"]

    def get_results(self, pred_results, detailed_output=False):
        pred_output = []
        for k, v in pred_results.items():
            ret = {
                'instr_id': k,
                'trajectory': v['path']
            }
            # object grounding
            if 'pred_objid' in v:
                ret.update({
                    'pred_objid': v['pred_objid'],
                    'pred_obj_direction': v['pred_obj_direction']
                })
            if detailed_output:
                ret.update({
                    'details': v['details']
                })
            pred_output.append(ret)

        return pred_output

    def rollout(self, *args, **kwargs):
        raise NotImplementedError
    
    def train(self, *args, **kwargs):
        raise NotImplementedError
    
    def validate(self, *args, **kwargs):
        raise NotImplementedError
