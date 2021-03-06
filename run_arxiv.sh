python train_continual.py --dim-proto 4 \
--dim-cls 4 \
--batch-size 100 \
--test-batch-size 5000 \
--devices 0 \
--num-epochs 90 \
--proto_est_epo 35 \
--proto_cls_epo 50 \
--data_name ogbn-arxiv \
--n_atoms_alloc 8000 8000 3000 \
--lr-decay 35 35 \
--n_nbs_per_hop 1 \
--n_AFE_a_select 1 \
--n_AFE_r_select 1 \
--atom_t 0.2