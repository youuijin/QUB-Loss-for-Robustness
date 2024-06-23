import pandas as pd

file_names = ['result', 'AutoAttack', 'PGD_Attack']
datasets = ['cifar10', 'cifar100', 'svhn'] 

for f in file_names:
    for dataset in datasets:
        file_name = f'results/{dataset}/csvs/{f}'
        df = pd.read_csv(f'{file_name}.csv')
        df.sort_values(by = 'model', inplace=True, ascending=True)
        df['model'] = df['model'].str.replace('/', '_')
        # df.sort_index() # 되돌리기

        # df.to_csv(f'{file_name}.csv', index=False)
        df.to_csv(f'{file_name}.csv', index=False)


# file_name = 'results/csvs/Bound'
# df = pd.read_csv(f'{file_name}.csv')
# df.sort_values(by = 'approx_metrics', inplace=True, ascending=True)

# # df.sort_index() # 되돌리기

# df.to_csv(f'{file_name}.csv', index=False)


# file_name = 'results/csvs/PGD_Attack'
# df = pd.read_csv(f'{file_name}.csv')
# df.sort_values(by = 'model', inplace=True, ascending=True)

# # df.sort_index() # 되돌리기

# df.to_csv(f'{file_name}.csv', index=False)



## GRAPH##

# results_df = pd.read_csv('results/csvs/result.csv')
# all_models = results_df['log_name'].unique()
# # print(all_models)

# final_df = pd.DataFrame(columns=['model', 'test_acc', 'test_adv_acc', 'PGD50', 'AA'])

# # .replace('/', '_')

# DATA = []
# for model in all_models:
#     aa_file = 'results/csvs/AutoAttack.csv'
#     pgd_file = 'results/csvs/PGD_Attack.csv'
#     test_acc = results_df.loc[results_df['log_name']==model, 'test_acc'].iloc[0]
#     test_adv_acc = results_df.loc[results_df['log_name']==model, 'test_adv_acc'].iloc[0]
#     attack_time = results_df.loc[results_df['log_name']==model, 'attack_time'].iloc[0]

#     model = model.replace('/', '_') + ".pt"
    
#     try:
#         aa_df = pd.read_csv(aa_file)
#         aa_test_value = aa_df.loc[aa_df['model'] == model, 'test_adv_acc'].iloc[0]
#     except IndexError:
#         aa_test_value = '-'
    
#     try:
#         pgd_df = pd.read_csv(pgd_file)
#         pgd_test_value = pgd_df.loc[pgd_df['model'] == model, 'test_adv_acc'].iloc[0]
#     except IndexError:
#         pgd_test_value = '-'
    
#     DATA.append({'model': model, 'test_acc': test_acc,'test_adv_acc':test_adv_acc, 'PGD50': pgd_test_value, 'AA': aa_test_value, 'attack_time':attack_time})
# final_df = pd.concat([final_df, pd.DataFrame(DATA)], ignore_index=True)

# # 결과 출력
# final_df.to_csv(f'results/csvs/best.csv', index=False)