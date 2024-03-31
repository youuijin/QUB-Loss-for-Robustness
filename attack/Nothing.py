from attack.AttackBase import Attack

class Nothing(Attack):
    def __init__(self, model):
        self.model = model

    def perturb(self, x, y):
        return x