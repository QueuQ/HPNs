python train_continual.py --dim-proto 2 \
--dim-cls 2 \
--batch-size 4000 \
--test-batch-size 20000 \
--devices 1 \
--num-epochs 50 \
--proto_est_epo 25 \
--proto_cls_epo 35 \
--data_name ogbn-products \
--n_nbs_per_hop 1 \
--n_AFE_a_select 1 \
--n_AFE_r_select 1 \
--n_AFE_a_alloc 23 \
--n_AFE_r_alloc 23 \
--atom_t 0.3

