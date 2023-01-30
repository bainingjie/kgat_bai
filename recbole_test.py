from recbole.quick_start import run_recbole
from torchcontrib.optim import SWA
# run_recbole(model='KGAT', dataset='ml-100k')

import argparse
from distutils.util import strtobool

# まずオブジェクト生成
parser = argparse.ArgumentParser()

# 引数設定
parser.add_argument("--seed", help="optional",type=int)
parser.add_argument("--aggr", help="optional")
parser.add_argument("--opti", help="optional")
parser.add_argument('--is_swa',type=strtobool)  
parser.add_argument('--is_momentum',type=strtobool)  

args = parser.parse_args()

# is_swa = True
# is_momentum = True
# if sys.argv[2]=="false":
#     is_swa = False,
# if sys.argv[3]=="false":
#     is_momentum = False,

parameter_dict = {
    'seed': args.seed,
    "is_swa":args.is_swa,
    "is_momentum":args.is_momentum,
    "aggregator_type":args.aggr,
    "learner":args.opti
}
config_file_list = ['kgat_new.yaml']
print(parameter_dict)
run_recbole(
    model='KGAT', dataset='ml-100k', config_file_list=config_file_list,config_dict=parameter_dict
)


