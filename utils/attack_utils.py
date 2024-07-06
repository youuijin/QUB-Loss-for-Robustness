from attack import AutoAttack, FastAttack, IterativeAttack

def set_attack(attack_name, model, eps, device, args, type='CE'):
    if attack_name in ['FGSM', 'FGSM_RS', 'FGSM_NR', 'FGSM_BR']:
        return FastAttack.FGSM(model, eps, args.a1, args.a2, initial=name_to_initial(attack_name), device=device)
    elif attack_name == 'FGSM_SDI':
        return FastAttack.FGSM_SDI(model, eps, args.a2, args.lr_att, device, args)
    elif attack_name == 'FGSM_CKPT':
        return FastAttack.FGSM_CKPT(model, eps, args.a1, args.a2, args.ckpt_init, args.ckpt_num, device) #TODO: uniform, 3
    elif attack_name == 'PGD_Linf':
        return IterativeAttack.PGDAttack(model, norm=args.norm, eps=eps, iter=args.iter, restart=args.restart, loss=type, device=device)
    elif attack_name == 'AA':
        return AutoAttack.AutoAttack(model, eps, args)
    else:
        raise ValueError("wrong type Attack")

def name_to_initial(attack_name):
    if attack_name == 'FGSM':
        return 'none'
    elif attack_name == 'FGSM_RS':
        return 'uniform'
    elif attack_name == 'FGSM_BR':
        return 'bernoulli'
    elif attack_name == 'FGSM_NR':
        return 'normal'
    else:
        return None