from modelscope.trainers.multi_modal import OFATrainer
from ..scst_criterion import SCSTCriterion

class OFATrainerWithSCST(OFATrainer):
    def __init__(self, 
                 use_scst: bool=False,
                 **kwargs):
        super().__init__(**kwargs)
        # only difference is about the 
        self.use_scst=use_scst
        if self.use_scst:
            self.criterion=SCSTCriterion(self.cfg.train.criterion)
