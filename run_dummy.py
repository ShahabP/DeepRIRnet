from deep_rirnet.dataset import make_dummy_dataset
from deep_rirnet.train import run_pretrain_and_finetune

if __name__ == "__main__":
    src_items = make_dummy_dataset(n=600)
    tgt_all = make_dummy_dataset(n=200)
    target_ft = tgt_all[:20]
    target_eval = tgt_all[20:]
    model, metrics = run_pretrain_and_finetune(src_items, target_ft, target_eval)
    print("Done. Metrics:", metrics)
