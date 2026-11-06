import torch

from synthetic_preliminary.train_inner_voice_v6 import local_boundary_loss, retain_records, retain_kl


def test_retention_records_and_kl():
    rows = retain_records(12, 32)
    assert len(rows) == 32 and len({r["prompt"] for r in rows}) == 32
    assert {r["kind"] for r in rows} == {
        "retain_math", "retain_schedule", "retain_inventory", "retain_location"
    }
    logits = torch.tensor([[[1.0, 2.0], [2.0, 1.0], [0.0, 3.0]]])
    labels = torch.tensor([[-100, 1, 0]])
    assert abs(float(retain_kl(logits, logits, labels))) < 1e-6


def test_local_loss_rewards_selective_eos():
    eos = 2
    logits = torch.tensor([[[5., 0., -5.], [0., 0., 5.]]])
    batch = {"shared_index": torch.tensor([0]), "boundary_index": torch.tensor([1]),
             "stop": torch.tensor([True])}
    good = local_boundary_loss(logits, batch, eos)
    bad = local_boundary_loss(logits.flip(1), batch, eos)
    assert good < bad
