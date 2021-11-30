
from transformers import Trainer
from transformers.optimization import get_scheduler
from ChildTuningOptimizer import ChildTuningAdamW

class ChildTuningFtrainer():
    def __init__(self, **kwargs,model):
        self.model=model
        self.weight_decay=kwargs.pop('weight_decay')
        self.adam_beta1 = kwargs.pop('adam_beta1')
        self.adam_beta2 = kwargs.pop('adam_beta2')
        self.adam_epsilon = kwargs.pop('adam_epsilon')
        self.learning_rate = kwargs.pop('learning_rate')
        
        self.reserve_p = kwargs.pop('reserve_p')
        self.mode = kwargs.pop('mode') # 'ChildTuning-D'
        self.lr_scheduler_type = kwargs.pop('lr_scheduler_type')
        self.warmup_steps = kwargs.pop('warmup_steps')
        self.num_training_steps = kwargs.pop('num_training_steps')
        
        self.optimizer = None
        self.lr_scheduler = None
        
        

    # model,weight_decay, adam_beta1, adam_beta2,adam_epsilon,learning_rate,reserve_p,mode,lr_scheduler_type,warmup_steps,num_training_steps
    def create_optimizer_and_scheduler(self):
        """
        Setup the optimizer and the learning rate scheduler.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            optimizer_cls = ChildTuningAdamW
            optimizer_kwargs = {
                "betas": (self.adam_beta1, self.adam_beta2),
                "eps": self.adam_epsilon,
            }
            optimizer_kwargs["lr"] = self.learning_rate
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, reserve_p=self.reserve_p, mode=self.mode, **optimizer_kwargs)

        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.lr_scheduler_type,
                self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.num_training_steps,
            )