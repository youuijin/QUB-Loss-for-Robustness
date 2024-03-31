from attack import Nothing, FGSM, PGD, rLF, Auto

def set_attack(attack_name, model, eps, args):
    initial = name_to_initial(attack_name)
    if attack_name == '':
        return Nothing.Nothing(model)
    elif 'FGSM' in attack_name:
        return FGSM.FGSM(model, eps, args.alpha, initial=initial)
    elif attack_name == 'PGD_Linf':
        return PGD.PGDAttack(model, norm=args.norm, eps=eps, iter=args.iter, restart=args.restart)
    elif 'rLF' in attack_name:
        return rLF.rLFAttack(model, eps, args.alpha, initial=initial)
    elif attack_name == 'AA':
        return Auto.AutoAttack(model, eps, args)
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
    elif attack_name in ['FGSM_LF']:
        return 'flat'
    else:
        return None