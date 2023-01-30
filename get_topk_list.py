import json

import numpy as np
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.quick_start import load_data_and_model
from recbole.utils import get_model
from recbole.utils.case_study import full_sort_topk
import pandas as pd
from tqdm.auto import tqdm
import itertools

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def output_score(model_file,output_file):
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=model_file
    )

    uid_list = []
    for batch_idx, batched_data in enumerate(test_data):
        interaction, row_idx, positive_u, positive_i = batched_data
        if(batch_idx==1):
            print("interaction",interaction)
        uid_list.append(interaction.user_id.numpy()[0])

    uid_list = list(set(uid_list))
    print("uid_list length",len(uid_list))

    ranked_list = []
    for uid in tqdm(uid_list):
        topk_score, topk_iid_list = full_sort_topk(
            [uid], model, test_data, k=10, device=config["device"]
        )
        # print("topk_iid_list",topk_iid_list.cpu())
        ranked_list += topk_iid_list.cpu()

    all_metrics_results = {}
    for uid, r_list in zip(uid_list, ranked_list):
        external_uid = dataset.id2token(dataset.uid_field, uid)
        all_metrics_results[external_uid] = {
            "predict_list_id": [v for v in dataset.id2token(dataset.iid_field, r_list)],
        }

    text = json.dumps(all_metrics_results, sort_keys=True, ensure_ascii=False, indent=2)
    with open(output_file, "w") as fh:
        fh.write(text)

def main():
    file1 = "KGAT-Jan-30-2023_17-16-18"
    file2 = "KGAT-Jan-30-2023_16-18-53"
    file3 = "KGAT-Jan-30-2023_17-20-40"

    # file4 = ""
    # file5 = "KGAT-Jan-26-2023_12-55-37"

    index_model={}
    output_score("saved/"+file1+".pth","score/"+file1+".json")
    output_score("saved/"+file2+".pth","score/"+file2+".json")
    output_score("saved/"+file3+".pth","score/"+file3+".json")
    # output_score("saved/"+file4+".pth","score/"+file4+".json")
    # output_score("saved/"+file5+".pth","score/"+file5+".json")
    filelist = ["score/"+file1+".json","score/"+file2+".json","score/"+file3+".json"
    # ,"score/"+file4+".json"
    # ,"score/"+file5+".json"
    ]
    model_results = {}
    for file in tqdm(filelist):
        _model = file.split("/")[-1].split(".")[0]
        try:
            _df = pd.read_json(file).T
            index_model[_model]=_df.index.values.tolist()
            model_results[_model] = _df
        except:
            print(f"{_model} read is failed")
    # modelのコンビネーションを用意する
    _models = model_results.keys() 
    combis = list(itertools.combinations(_models, 2))
    model_similarities = []

    # check if test user ids are the same
    # print(list(_models)[0])
    for uid1, uid2 in zip(index_model[list(_models)[0]], index_model[list(_models)[1]]):
        if(uid1 != uid2):
            print("不一致が発生した。",uid1,uid2)
    

    for c in tqdm(combis):
        model1 = c[0]
        model2 = c[1]
        model1_result = model_results[model1]
        model2_result = model_results[model2] 
        print("model1_result",model1_result.columns)
        # print("model1_result _ predict_list_id",model1_result["predict_list_id"])
        model1_predict_list = model1_result["predict_list_id"].values
        # print("model1_predict_list ",model1_predict_list )
        model2_predict_list = model2_result["predict_list_id"].values

        sims = []
        for m1_preds, m2_preds in zip(model1_predict_list, model2_predict_list):
            _sim = len(set(m1_preds) & set(m2_preds)) / len(m1_preds)
            sims.append(_sim)
        similarity = np.mean(sims)
        model_similarities.append([model1, model2, similarity])
        result = pd.DataFrame(
            model_similarities, columns=["source_model", "dest_model", "similarity"]
        )
        result.to_csv(file1+"@"+file2+"@"+file3
        # +"@"+file4
        # +"@"+file5
        +".csv", index=False)

if __name__ == "__main__":
    main()





