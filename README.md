
シンプソン係数
//0.897
aggregator_type: 'bi'(default)
saved/KGAT-Jan-15-2023_17-53-21.pth
Sun 15 Jan 2023 17:57:53 INFO  test result: OrderedDict([('recall@10', 0.2364), ('mrr@10', 0.46), ('ndcg@10', 0.2786), ('hit@10', 0.754), ('precision@10', 0.1903)])

saved/KGAT-Jan-15-2023_19-36-57.pth


aggregator_type: 'gcn'
//0.847
saved/KGAT-Jan-19-2023_14-13-40.pth
saved/KGAT-Jan-19-2023_20-18-13.pth
saved/KGAT-Jan-19-2023_20-57-46.pth //なしで

aggregator_type: 'graphsage'
//0.532
saved/KGAT-Jan-19-2023_14-37-30.pth
saved/KGAT-Jan-20-2023_00-31-04.pth
saved/KGAT-Jan-25-2023_23-24-05.pth
INFO  test result: OrderedDict([('recall@10', 0.2339), ('mrr@10', 0.4509), ('ndcg@10', 0.2738), ('hit@10', 0.7625), ('precision@10', 0.1881)])
saved/KGAT-Jan-26-2023_07-29-44.pth
INFO  test result: OrderedDict([('recall@10', 0.2411), ('mrr@10', 0.448), ('ndcg@10', 0.2776), ('hit@10', 0.7678), ('precision@10', 0.1923)])

SEED 2021 vs 2020(default)
bi
//0.465

/work/gu14/k36095/kgat_bai/log/KGAT/KGAT-ml-100k-Jan-20-2023_15-53-29-967b9c.log
saved/KGAT-Jan-20-2023_15-53-38.pth


SEED 2021 vs 2020(default)
gcn
//0.44533898305084746
Saving current: saved/KGAT-Jan-25-2023_22-47-07.pth
/work/gu14/k36095/kgat_bai/log/KGAT/KGAT-ml-100k-Jan-25-2023_22-46-57-272357.log
OrderedDict([('recall@10', 0.2588), ('mrr@10', 0.4719), ('ndcg@10', 0.2932), ('hit@10', 0.7858), ('precision@10', 0.1994)])

SEED 2021 vs 2020(default)
graph sage
//0.465
saved/KGAT-Jan-25-2023_23-05-36.pth
test result: OrderedDict([('recall@10', 0.2579), ('mrr@10', 0.4579), ('ndcg@10', 0.2878), ('hit@10', 0.7784), ('precision@10', 0.1975)])

negative sampling
distribution vs ratio


=========================-

追加
optimizer.swap_swa_sgd()
self.optimizer.defaults=self.optimizer.optimizer.defaults