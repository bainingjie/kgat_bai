from recbole.quick_start import run_recbole

# run_recbole(model='KGAT', dataset='ml-100k')

config_file_list = ['kgat_new.yaml']
run_recbole(model='KGAT', dataset='ml-100k', config_file_list=config_file_list)
