from attack import AutoAttack, FastAttack, IterativeAttack

def set_attack(attack_name, model, eps, args, type='CE'):
    initial = name_to_initial(attack_name)
    if 'FGSM' in attack_name:
        return FastAttack.FGSM(model, eps, args.a1, args.a2, initial=initial)
    elif attack_name == 'PGD_Linf':
        return IterativeAttack.PGDAttack(model, norm=args.norm, eps=eps, iter=args.iter, restart=args.restart, loss=type)
    elif attack_name == 'AA':
        return AutoAttack.AutoAttack(model, eps, args)
    else:
        raise ValueError("wrong type Attack")

def name_to_initial(attack_name):
    if attack_name in ['FGSM', 'rLF']:
        return 'none'
    elif attack_name in ['FGSM_RS', 'rLF_RS']:
        return 'uniform'
    elif attack_name in ['FGSM_BR', 'rLF_BR']:
        return 'bernoulli'
    elif attack_name in ['FGSM_NR', 'rLF_NR']:
        return 'normal'
    else:
        return None